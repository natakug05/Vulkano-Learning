use std::{error::Error, sync::Arc};

use image::{ImageBuffer, Rgba};
use vulkano::{
    Validated, VulkanLibrary,
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
        CopyImageToBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, physical::PhysicalDevice,
    },
    format::{ClearColorValue, Format},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderModule,
    sync::{self, GpuFuture},
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/multiply.comp",
    }
}

mod mandelbrot_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/mandelbrot.comp",
    }
}

struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
}

// Dummy struct
#[derive(BufferContents)]
#[repr(C)]
struct MyStruct {
    a: u32,
    b: u32,
}

fn allocate_struct_data_buffer(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    data: MyStruct,
) -> Result<Subbuffer<MyStruct>, Validated<AllocateBufferError>> {
    Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )
}

fn allocate_simple_data_buffer(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    data: i32,
) -> Result<Subbuffer<i32>, Validated<AllocateBufferError>> {
    Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )
}

fn initialize_device() -> Result<Arc<PhysicalDevice>, Box<dyn Error>> {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let vulkano_instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap_or_else(|err| panic!("failed to create instance: {:?}", err));
    vulkano_instance
        .enumerate_physical_devices()?
        .next()
        .ok_or("no devices".into())
}

fn simple_allocation(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let _simple_buffer = allocate_simple_data_buffer(&ctx.memory_allocator, 12)?;
    let _struct_buffer =
        allocate_struct_data_buffer(&ctx.memory_allocator, MyStruct { a: 5, b: 69 })?;

    let iter = (0..128).map(|_| 5u8);
    let buffer = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        iter,
    )
    .unwrap();

    let mut content = buffer.write().unwrap();
    content[12] = 83;
    content[7] = 3;

    Ok(())
}

// Chapter 3 command buffer intro
fn command_buffer_copy_with_future(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let source_content = 0..64;
    let source = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    )?;

    let destination_content = (0..64).map(|_| 0);
    let destination = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_content,
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let src_content = source.read().unwrap();
    let destination_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);

    println!("Everything succeeded!");

    Ok(())
}

// Chapter 4/5 combined shader pipeline helper
fn create_compute_pipeline(
    device: &Arc<Device>,
    shader: Arc<ShaderModule>,
) -> Result<Arc<ComputePipeline>, Box<dyn std::error::Error>> {
    let entry = shader
        .entry_point("main")
        .ok_or("shader has no `main` entry point")?;
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())?,
    )?;
    Ok(ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?)
}

// Chapter 4 multiply shader
fn perform_gpu_compute(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )?;

    let compute_pipeline = create_compute_pipeline(&ctx.device, cs::load(ctx.device.clone())?)?;

    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &ctx.descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )
    .unwrap();

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let work_group_counts = [1024, 1, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .unwrap()
        .dispatch(work_group_counts)
        .unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();

    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");

    Ok(())
}

// Chapter 5 generate a blue PNG from the GPU
fn create_image(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;

    let buf = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        })?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))?;

    let command_buffer = builder.build()?;

    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    future.wait(None)?;

    let buffer_content = buf.read()?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..])
        .ok_or("Couldn't convert image content to PNG")?;
    image.save("artifacts/image.png")?;

    println!("Everything succeeded!");

    Ok(())
}

// Chapter 5 cont. with fractal shader
fn draw_fractal(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;

    let view = ImageView::new_default(image.clone())?;
    let compute_pipeline =
        create_compute_pipeline(&ctx.device, mandelbrot_cs::load(ctx.device.clone())?)?;
    let layout = compute_pipeline
        .layout()
        .set_layouts()
        .get(0)
        .ok_or("couldn't load compute layout")?;
    let set = PersistentDescriptorSet::new(
        &ctx.descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
        [],
    )?;

    let buf = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder
        .bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )?
        .dispatch([1024 / 8, 1024 / 8, 1])?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))?;

    let command_buffer = builder.build()?;

    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    future.wait(None)?;

    let buffer_content = buf.read()?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..])
        .ok_or("could not convert image to png")?;
    image.save("artifacts/fractal.png")?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let physical_device = initialize_device()?;
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .ok_or("no device queue found")? as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )?;

    let queue = queues.next().ok_or("no queue")?;
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let ctx = GpuContext {
        device,
        queue,
        memory_allocator,
        command_buffer_allocator,
        descriptor_set_allocator,
    };

    simple_allocation(&ctx)?;
    command_buffer_copy_with_future(&ctx)?;
    perform_gpu_compute(&ctx)?;
    create_image(&ctx)?;
    draw_fractal(&ctx)?;

    Ok(())
}
