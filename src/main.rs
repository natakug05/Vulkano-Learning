use std::{error::Error, sync::Arc};

use image::{ImageBuffer, Rgba};
use vulkano::{
    Validated, VulkanLibrary, buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    }, command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
        CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    }, descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    }, device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    }, format::{ClearColorValue, Format}, image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::ShaderModule, swapchain::{Surface, Swapchain, SwapchainCreateInfo}, sync::{self, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
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

mod triangle {
    vulkano_shaders::shader! {
        shaders: {
            vs: { ty: "vertex", path: "src/shaders/triangle.vert" },
            fs: { ty: "fragment", path: "src/shaders/triangle.frag" },
        }
    }
}

struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
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

fn select_physical_device(
    vulkan_instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    vulkan_instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no device available")
}

fn initialize_vulkan(event_loop: &EventLoop<()>) -> Result<Arc<Instance>, Box<dyn Error>> {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let required_extensions = Surface::required_extensions(&event_loop);
    let vulkan_instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap_or_else(|err| panic!("failed to create instance: {:?}", err));

    Ok(vulkan_instance)
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

    println!("Created command buffer successfully!");

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

    println!("Performed GPU multiplication compute shader successfully!");

    Ok(())
}

// Helper function for the many calls we saw to image creation
fn create_image(
    ctx: &GpuContext,
    usage: ImageUsage,
) -> Result<Arc<Image>, Box<dyn std::error::Error>> {
    Ok(Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?)
}

// Chapter 5 generate a blue PNG from the GPU
fn draw_basic_image(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let image = create_image(&ctx, ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC)?;

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

    println!("Drew basic blue image successfully!");

    Ok(())
}

// Chapter 5 cont. with fractal shader
fn draw_fractal(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let image = create_image(&ctx, ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC)?;

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

    println!("Drew mandelbrot set fractal image successfully!");

    Ok(())
}

fn create_graphics_pipeline(
    ctx: &GpuContext,
    render_pass: &Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>, Box<dyn std::error::Error>> {
    // A Vulkan shader can in theory contain multiple entry points, so we have to specify
    // which one.
    let vs = triangle::load_vs(ctx.device.clone())?
        .entry_point("main")
        .ok_or("vertex shader has no `main` entry point")?;
    let fs = triangle::load_fs(ctx.device.clone())?
        .entry_point("main")
        .ok_or("fragment shader has no `main` entry point")?;

    let vertex_input_state = MyVertex::per_vertex().definition(&vs.info().input_interface)?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        ctx.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(ctx.device.clone())?,
    )?;

    let subpass =
        Subpass::from(render_pass.clone(), 0).ok_or("Could not create subpass on render pass")?;

    Ok(GraphicsPipeline::new(
        ctx.device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )?)
}

// Chapter 6 drawing a triangle using the pipeline
fn draw_triangle(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    let vertex1 = MyVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = MyVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = MyVertex {
        position: [0.5, -0.25],
    };

    let vertex_buffer = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )?;

    let render_pass = vulkano::single_pass_renderpass!(
        ctx.device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )?;

    let image = create_image(
        &ctx,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
    )?;
    let view = ImageView::new_default(image.clone())?;
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )?;

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [1024.0, 1024.0],
        depth_range: 0.0..=1.0,
    };

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

    let pipeline = create_graphics_pipeline(&ctx, &render_pass, viewport)?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?
        .bind_pipeline_graphics(pipeline.clone())?
        .bind_vertex_buffers(0, vertex_buffer.clone())?
        .draw(3, 1, 0, 0)?
        .end_render_pass(SubpassEndInfo::default())?
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
        .ok_or("Couldn't convert image to png")?;
    image.save("artifacts/triangle.png")?;

    println!("Drew triangle successfully!");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let vulkan_instance = initialize_vulkan(&event_loop)?;
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let surface = Surface::from_window(vulkan_instance.clone(), window.clone())?;
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) =
        select_physical_device(&vulkan_instance, &surface, &device_extensions);

    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("failed to get surface capabilities");

    let dimensions = window.inner_size();
    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .ok_or("could not get composite_alpha for window")?;
    // Eww why doesn't the tutorial explain the zeroth member of zeroth position in this vector...
    // Assumption is we just want anything and the first Format will do.
    let image_format = physical_device
        .surface_formats(&surface, Default::default())?[0].0;

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )?;

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
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
    draw_basic_image(&ctx)?;
    draw_fractal(&ctx)?;
    draw_triangle(&ctx)?;

    event_loop.run(|event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => {}
    });
}
