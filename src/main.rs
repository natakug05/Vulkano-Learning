use std::{error::Error, sync::Arc};

use vulkano::{
    Validated, VulkanLibrary, buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    }, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    }}, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator}, device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, physical::PhysicalDevice}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo}, sync::{self, GpuFuture},
};

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/multiply.comp",
    }
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

fn simple_allocation(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _simple_buffer = allocate_simple_data_buffer(&memory_allocator, 12)?;
    let _struct_buffer = allocate_struct_data_buffer(&memory_allocator, MyStruct { a: 5, b: 69 })?;

    let iter = (0..128).map(|_| 5u8);
    let buffer = Buffer::from_iter(
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
        iter,
    )
    .unwrap();

    let mut content = buffer.write().unwrap();
    content[12] = 83;
    content[7] = 3;

    Ok(())
}

fn command_buffer_copy_with_future(
    device: &Arc<Device>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    queue_family_index: u32,
    queue: &Arc<Queue>,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_content = 0..64;
    let source = Buffer::from_iter(
        memory_allocator.clone(),
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
        memory_allocator.clone(),
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

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
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

fn perform_gpu_compute(
    device: &Arc<Device>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    queue: &Arc<Queue>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
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

    let shader = cs::load(device.clone())?;
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )
    .unwrap();

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
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

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
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

    let (gfx_device, mut queues) = Device::new(
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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(gfx_device.clone()));

    simple_allocation(&memory_allocator)?;
    command_buffer_copy_with_future(&gfx_device, &memory_allocator, queue_family_index, &queue)?;
    perform_gpu_compute(&gfx_device, &memory_allocator, &queue)?;

    Ok(())
}
