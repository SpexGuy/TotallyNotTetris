const std = @import("std");
usingnamespace @import("vk_core.zig");

pub const struct_VkSamplerYcbcrConversion_T = @OpaqueType();
pub const SamplerYcbcrConversion = ?*struct_VkSamplerYcbcrConversion_T;
pub const struct_VkDescriptorUpdateTemplate_T = @OpaqueType();
pub const DescriptorUpdateTemplate = ?*struct_VkDescriptorUpdateTemplate_T;

pub const PointClippingBehavior = extern enum {
    ALL_CLIP_PLANES = 0,
    USER_CLIP_PLANES_ONLY = 1,
    //ALL_CLIP_PLANES_KHR = 0,
    //USER_CLIP_PLANES_ONLY_KHR = 1,
};

pub const TessellationDomainOrigin = extern enum {
    UPPER_LEFT = 0,
    LOWER_LEFT = 1,
    //UPPER_LEFT_KHR = 0,
    //LOWER_LEFT_KHR = 1,
};

pub const SamplerYcbcrModelConversion = extern enum {
    RGB_IDENTITY = 0,
    YCBCR_IDENTITY = 1,
    YCBCR_709 = 2,
    YCBCR_601 = 3,
    YCBCR_2020 = 4,
    //RGB_IDENTITY_KHR = 0,
    //YCBCR_IDENTITY_KHR = 1,
    //YCBCR_709_KHR = 2,
    //YCBCR_601_KHR = 3,
    //YCBCR_2020_KHR = 4,
};

pub const SamplerYcbcrRange = extern enum {
    ITU_FULL = 0,
    ITU_NARROW = 1,
    //ITU_FULL_KHR = 0,
    //ITU_NARROW_KHR = 1,
};

pub const ChromaLocation = extern enum {
    COSITED_EVEN = 0,
    MIDPOINT = 1,
    //COSITED_EVEN_KHR = 0,
    //MIDPOINT_KHR = 1,
};

pub const DescriptorUpdateTemplateType = extern enum {
    DESCRIPTOR_SET = 0,
    PUSH_DESCRIPTORS_KHR = 1,
    //DESCRIPTOR_SET_KHR = 0,
};

pub const SubgroupFeatureFlags = Flags;
pub const SubgroupFeatureFlagBits = struct {
    pub const BASIC_BIT = 1;
    pub const VOTE_BIT = 2;
    pub const ARITHMETIC_BIT = 4;
    pub const BALLOT_BIT = 8;
    pub const SHUFFLE_BIT = 16;
    pub const SHUFFLE_RELATIVE_BIT = 32;
    pub const CLUSTERED_BIT = 64;
    pub const QUAD_BIT = 128;
    pub const PARTITIONED_BIT_NV = 256;
};

pub const PeerMemoryFeatureFlags = Flags;
pub const PeerMemoryFeatureFlagBits = struct {
    pub const COPY_SRC_BIT = 1;
    pub const COPY_DST_BIT = 2;
    pub const GENERIC_SRC_BIT = 4;
    pub const GENERIC_DST_BIT = 8;
    pub const COPY_SRC_BIT_KHR = 1;
    pub const COPY_DST_BIT_KHR = 2;
    pub const GENERIC_SRC_BIT_KHR = 4;
    pub const GENERIC_DST_BIT_KHR = 8;
};

pub const MemoryAllocateFlags = Flags;
pub const MemoryAllocateFlagBits = struct {
    pub const DEVICE_MASK_BIT = 1;
    pub const DEVICE_MASK_BIT_KHR = 1;
};

pub const CommandPoolTrimFlags = Flags;
pub const DescriptorUpdateTemplateCreateFlags = Flags;

pub const ExternalMemoryHandleTypeFlags = Flags;
pub const ExternalMemoryHandleTypeFlagBits = struct {
    pub const OPAQUE_FD_BIT = 1;
    pub const OPAQUE_WIN32_BIT = 2;
    pub const OPAQUE_WIN32_KMT_BIT = 4;
    pub const D3D11_TEXTURE_BIT = 8;
    pub const D3D11_TEXTURE_KMT_BIT = 16;
    pub const D3D12_HEAP_BIT = 32;
    pub const D3D12_RESOURCE_BIT = 64;
    pub const DMA_BUF_BIT_EXT = 512;
    pub const ANDROID_HARDWARE_BUFFER_BIT_ANDROID = 1024;
    pub const HOST_ALLOCATION_BIT_EXT = 128;
    pub const HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT = 256;
    pub const OPAQUE_FD_BIT_KHR = 1;
    pub const OPAQUE_WIN32_BIT_KHR = 2;
    pub const OPAQUE_WIN32_KMT_BIT_KHR = 4;
    pub const D3D11_TEXTURE_BIT_KHR = 8;
    pub const D3D11_TEXTURE_KMT_BIT_KHR = 16;
    pub const D3D12_HEAP_BIT_KHR = 32;
    pub const D3D12_RESOURCE_BIT_KHR = 64;
};

pub const ExternalMemoryFeatureFlags = Flags;
pub const ExternalMemoryFeatureFlagBits = struct {
    pub const DEDICATED_ONLY_BIT = 1;
    pub const EXPORTABLE_BIT = 2;
    pub const IMPORTABLE_BIT = 4;
    pub const DEDICATED_ONLY_BIT_KHR = 1;
    pub const EXPORTABLE_BIT_KHR = 2;
    pub const IMPORTABLE_BIT_KHR = 4;
};

pub const ExternalFenceHandleTypeFlags = Flags;
pub const ExternalFenceHandleTypeFlagBits = struct {
    pub const OPAQUE_FD_BIT = 1;
    pub const OPAQUE_WIN32_BIT = 2;
    pub const OPAQUE_WIN32_KMT_BIT = 4;
    pub const SYNC_FD_BIT = 8;
    pub const OPAQUE_FD_BIT_KHR = 1;
    pub const OPAQUE_WIN32_BIT_KHR = 2;
    pub const OPAQUE_WIN32_KMT_BIT_KHR = 4;
    pub const SYNC_FD_BIT_KHR = 8;
};

pub const ExternalFenceFeatureFlags = Flags;
pub const ExternalFenceFeatureFlagBits = struct {
    pub const EXPORTABLE_BIT = 1;
    pub const IMPORTABLE_BIT = 2;
    pub const EXPORTABLE_BIT_KHR = 1;
    pub const IMPORTABLE_BIT_KHR = 2;
};

pub const FenceImportFlags = Flags;
pub const FenceImportFlagBits = struct {
    pub const TEMPORARY_BIT = 1;
    pub const TEMPORARY_BIT_KHR = 1;
};

pub const SemaphoreImportFlags = Flags;
pub const SemaphoreImportFlagBits = struct {
    pub const TEMPORARY_BIT = 1;
    pub const TEMPORARY_BIT_KHR = 1;
};

pub const ExternalSemaphoreHandleTypeFlags = Flags;
pub const ExternalSemaphoreHandleTypeFlagBits = struct {
    pub const OPAQUE_FD_BIT = 1;
    pub const OPAQUE_WIN32_BIT = 2;
    pub const OPAQUE_WIN32_KMT_BIT = 4;
    pub const D3D12_FENCE_BIT = 8;
    pub const SYNC_FD_BIT = 16;
    pub const OPAQUE_FD_BIT_KHR = 1;
    pub const OPAQUE_WIN32_BIT_KHR = 2;
    pub const OPAQUE_WIN32_KMT_BIT_KHR = 4;
    pub const D3D12_FENCE_BIT_KHR = 8;
    pub const SYNC_FD_BIT_KHR = 16;
};

pub const ExternalSemaphoreFeatureFlags = Flags;
pub const ExternalSemaphoreFeatureFlagBits = struct {
    pub const EXPORTABLE_BIT = 1;
    pub const IMPORTABLE_BIT = 2;
    pub const EXPORTABLE_BIT_KHR = 1;
    pub const IMPORTABLE_BIT_KHR = 2;
};





pub const PhysicalDeviceSubgroupProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    pNext: ?*c_void = null,
    subgroupSize: u32,
    supportedStages: ShaderStageFlags,
    supportedOperations: SubgroupFeatureFlags,
    quadOperationsInAllStages: Bool32,
};

pub const BindBufferMemoryInfo = extern struct {
    sType: StructureType = .BIND_BUFFER_MEMORY_INFO,
    pNext: ?*const c_void = null,
    buffer: Buffer,
    memory: DeviceMemory,
    memoryOffset: DeviceSize,
};

pub const BindImageMemoryInfo = extern struct {
    sType: StructureType = .BIND_IMAGE_MEMORY_INFO,
    pNext: ?*const c_void = null,
    image: Image,
    memory: DeviceMemory,
    memoryOffset: DeviceSize,
};

pub const PhysicalDevice16BitStorageFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    pNext: ?*c_void = null,
    storageBuffer16BitAccess: Bool32,
    uniformAndStorageBuffer16BitAccess: Bool32,
    storagePushConstant16: Bool32,
    storageInputOutput16: Bool32,
};

pub const MemoryDedicatedRequirements = extern struct {
    sType: StructureType = .MEMORY_DEDICATED_REQUIREMENTS,
    pNext: ?*c_void = null,
    prefersDedicatedAllocation: Bool32,
    requiresDedicatedAllocation: Bool32,
};

pub const MemoryDedicatedAllocateInfo = extern struct {
    sType: StructureType = .MEMORY_DEDICATED_ALLOCATE_INFO,
    pNext: ?*const c_void = null,
    image: Image,
    buffer: Buffer,
};

pub const MemoryAllocateFlagsInfo = extern struct {
    sType: StructureType = .MEMORY_ALLOCATE_FLAGS_INFO,
    pNext: ?*const c_void = null,
    flags: MemoryAllocateFlags,
    deviceMask: u32,
};

pub const DeviceGroupRenderPassBeginInfo = extern struct {
    sType: StructureType = .DEVICE_GROUP_RENDER_PASS_BEGIN_INFO,
    pNext: ?*const c_void = null,
    deviceMask: u32,
    deviceRenderAreaCount: u32,
    pDeviceRenderAreas: ?[*]const Rect2D,
};

pub const DeviceGroupCommandBufferBeginInfo = extern struct {
    sType: StructureType = .DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const c_void = null,
    deviceMask: u32,
};

pub const DeviceGroupSubmitInfo = extern struct {
    sType: StructureType = .DEVICE_GROUP_SUBMIT_INFO,
    pNext: ?*const c_void = null,
    waitSemaphoreCount: u32,
    pWaitSemaphoreDeviceIndices: ?[*]const u32,
    commandBufferCount: u32,
    pCommandBufferDeviceMasks: ?[*]const u32,
    signalSemaphoreCount: u32,
    pSignalSemaphoreDeviceIndices: ?[*]const u32,
};

pub const DeviceGroupBindSparseInfo = extern struct {
    sType: StructureType = .DEVICE_GROUP_BIND_SPARSE_INFO,
    pNext: ?*const c_void = null,
    resourceDeviceIndex: u32,
    memoryDeviceIndex: u32,
};

pub const BindBufferMemoryDeviceGroupInfo = extern struct {
    sType: StructureType = .BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO,
    pNext: ?*const c_void = null,
    deviceIndexCount: u32,
    pDeviceIndices: ?[*]const u32,
};

pub const BindImageMemoryDeviceGroupInfo = extern struct {
    sType: StructureType = .BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO,
    pNext: ?*const c_void = null,
    deviceIndexCount: u32,
    pDeviceIndices: ?[*]const u32,
    splitInstanceBindRegionCount: u32,
    pSplitInstanceBindRegions: ?[*]const Rect2D,
};

pub const PhysicalDeviceGroupProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_GROUP_PROPERTIES,
    pNext: ?*c_void = null,
    physicalDeviceCount: u32,
    physicalDevices: [32]PhysicalDevice,
    subsetAllocation: Bool32,
};

pub const DeviceGroupDeviceCreateInfo = extern struct {
    sType: StructureType = .DEVICE_GROUP_DEVICE_CREATE_INFO,
    pNext: ?*const c_void = null,
    physicalDeviceCount: u32,
    pPhysicalDevices: ?[*]const PhysicalDevice,
};

pub const BufferMemoryRequirementsInfo2 = extern struct {
    sType: StructureType = .BUFFER_MEMORY_REQUIREMENTS_INFO_2,
    pNext: ?*const c_void = null,
    buffer: Buffer,
};

pub const ImageMemoryRequirementsInfo2 = extern struct {
    sType: StructureType = .IMAGE_MEMORY_REQUIREMENTS_INFO_2,
    pNext: ?*const c_void = null,
    image: Image,
};

pub const ImageSparseMemoryRequirementsInfo2 = extern struct {
    sType: StructureType = .IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
    pNext: ?*const c_void = null,
    image: Image,
};

pub const MemoryRequirements2 = extern struct {
    sType: StructureType = .MEMORY_REQUIREMENTS_2,
    pNext: ?*c_void = null,
    memoryRequirements: MemoryRequirements,
};

pub const SparseImageMemoryRequirements2 = extern struct {
    sType: StructureType = .SPARSE_IMAGE_MEMORY_REQUIREMENTS_2,
    pNext: ?*c_void = null,
    memoryRequirements: SparseImageMemoryRequirements,
};

pub const PhysicalDeviceFeatures2 = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_FEATURES_2,
    pNext: ?*c_void = null,
    features: PhysicalDeviceFeatures,
};

pub const PhysicalDeviceProperties2 = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_PROPERTIES_2,
    pNext: ?*c_void = null,
    properties: PhysicalDeviceProperties,
};

pub const FormatProperties2 = extern struct {
    sType: StructureType = .FORMAT_PROPERTIES_2,
    pNext: ?*c_void = null,
    formatProperties: FormatProperties,
};

pub const ImageFormatProperties2 = extern struct {
    sType: StructureType = .IMAGE_FORMAT_PROPERTIES_2,
    pNext: ?*c_void = null,
    imageFormatProperties: ImageFormatProperties,
};

pub const PhysicalDeviceImageFormatInfo2 = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    pNext: ?*const c_void = null,
    format: Format,
    type: ImageType,
    tiling: ImageTiling,
    usage: ImageUsageFlags,
    flags: ImageCreateFlags,
};

pub const QueueFamilyProperties2 = extern struct {
    sType: StructureType = .QUEUE_FAMILY_PROPERTIES_2,
    pNext: ?*c_void = null,
    queueFamilyProperties: QueueFamilyProperties,
};

pub const PhysicalDeviceMemoryProperties2 = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    pNext: ?*c_void = null,
    memoryProperties: PhysicalDeviceMemoryProperties,
};

pub const SparseImageFormatProperties2 = extern struct {
    sType: StructureType = .SPARSE_IMAGE_FORMAT_PROPERTIES_2,
    pNext: ?*c_void = null,
    properties: SparseImageFormatProperties,
};

pub const PhysicalDeviceSparseImageFormatInfo2 = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
    pNext: ?*const c_void = null,
    format: Format,
    type: ImageType,
    samples: SampleCountFlags,
    usage: ImageUsageFlags,
    tiling: ImageTiling,
};

pub const PhysicalDevicePointClippingProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES,
    pNext: ?*c_void = null,
    pointClippingBehavior: PointClippingBehavior,
};

pub const InputAttachmentAspectReference = extern struct {
    subpass: u32,
    inputAttachmentIndex: u32,
    aspectMask: ImageAspectFlags,
};

pub const RenderPassInputAttachmentAspectCreateInfo = extern struct {
    sType: StructureType = .RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO,
    pNext: ?*const c_void = null,
    aspectReferenceCount: u32,
    pAspectReferences: ?[*]const InputAttachmentAspectReference,
};

pub const ImageViewUsageCreateInfo = extern struct {
    sType: StructureType = .IMAGE_VIEW_USAGE_CREATE_INFO,
    pNext: ?*const c_void = null,
    usage: ImageUsageFlags,
};

pub const PipelineTessellationDomainOriginStateCreateInfo = extern struct {
    sType: StructureType = .PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
    pNext: ?*const c_void = null,
    domainOrigin: TessellationDomainOrigin,
};

pub const RenderPassMultiviewCreateInfo = extern struct {
    sType: StructureType = .RENDER_PASS_MULTIVIEW_CREATE_INFO,
    pNext: ?*const c_void = null,
    subpassCount: u32,
    pViewMasks: ?[*]const u32,
    dependencyCount: u32,
    pViewOffsets: ?[*]const i32,
    correlationMaskCount: u32,
    pCorrelationMasks: ?[*]const u32,
};

pub const PhysicalDeviceMultiviewFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
    pNext: ?*c_void = null,
    multiview: Bool32,
    multiviewGeometryShader: Bool32,
    multiviewTessellationShader: Bool32,
};

pub const PhysicalDeviceMultiviewProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES,
    pNext: ?*c_void = null,
    maxMultiviewViewCount: u32,
    maxMultiviewInstanceIndex: u32,
};

pub const PhysicalDeviceVariablePointerFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES,
    pNext: ?*c_void = null,
    variablePointersStorageBuffer: Bool32,
    variablePointers: Bool32,
};

pub const PhysicalDeviceProtectedMemoryFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
    pNext: ?*c_void = null,
    protectedMemory: Bool32,
};

pub const PhysicalDeviceProtectedMemoryProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES,
    pNext: ?*c_void = null,
    protectedNoFault: Bool32,
};

pub const DeviceQueueInfo2 = extern struct {
    sType: StructureType = .DEVICE_QUEUE_INFO_2,
    pNext: ?*const c_void = null,
    flags: DeviceQueueCreateFlags,
    queueFamilyIndex: u32,
    queueIndex: u32,
};

pub const ProtectedSubmitInfo = extern struct {
    sType: StructureType = .PROTECTED_SUBMIT_INFO,
    pNext: ?*const c_void = null,
    protectedSubmit: Bool32,
};

pub const SamplerYcbcrConversionCreateInfo = extern struct {
    sType: StructureType = .SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
    pNext: ?*const c_void = null,
    format: Format,
    ycbcrModel: SamplerYcbcrModelConversion,
    ycbcrRange: SamplerYcbcrRange,
    components: ComponentMapping,
    xChromaOffset: ChromaLocation,
    yChromaOffset: ChromaLocation,
    chromaFilter: Filter,
    forceExplicitReconstruction: Bool32,
};

pub const SamplerYcbcrConversionInfo = extern struct {
    sType: StructureType = .SAMPLER_YCBCR_CONVERSION_INFO,
    pNext: ?*const c_void = null,
    conversion: SamplerYcbcrConversion,
};

pub const BindImagePlaneMemoryInfo = extern struct {
    sType: StructureType = .BIND_IMAGE_PLANE_MEMORY_INFO,
    pNext: ?*const c_void = null,
    planeAspect: ImageAspectFlags,
};

pub const ImagePlaneMemoryRequirementsInfo = extern struct {
    sType: StructureType = .IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
    pNext: ?*const c_void = null,
    planeAspect: ImageAspectFlags,
};

pub const PhysicalDeviceSamplerYcbcrConversionFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
    pNext: ?*c_void = null,
    samplerYcbcrConversion: Bool32,
};

pub const SamplerYcbcrConversionImageFormatProperties = extern struct {
    sType: StructureType = .SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
    pNext: ?*c_void = null,
    combinedImageSamplerDescriptorCount: u32,
};

pub const DescriptorUpdateTemplateEntry = extern struct {
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: DescriptorType,
    offset: usize,
    stride: usize,
};

pub const DescriptorUpdateTemplateCreateInfo = extern struct {
    sType: StructureType = .DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
    pNext: ?*c_void = null,
    flags: DescriptorUpdateTemplateCreateFlags,
    descriptorUpdateEntryCount: u32,
    pDescriptorUpdateEntries: ?[*]const DescriptorUpdateTemplateEntry,
    templateType: DescriptorUpdateTemplateType,
    descriptorSetLayout: DescriptorSetLayout,
    pipelineBindPoint: PipelineBindPoint,
    pipelineLayout: PipelineLayout,
    set: u32,
};

pub const ExternalMemoryProperties = extern struct {
    externalMemoryFeatures: ExternalMemoryFeatureFlags,
    exportFromImportedHandleTypes: ExternalMemoryHandleTypeFlags,
    compatibleHandleTypes: ExternalMemoryHandleTypeFlags,
};

pub const PhysicalDeviceExternalImageFormatInfo = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
    pNext: ?*const c_void = null,
    handleType: ExternalMemoryHandleTypeFlags,
};

pub const ExternalImageFormatProperties = extern struct {
    sType: StructureType = .EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    pNext: ?*c_void = null,
    externalMemoryProperties: ExternalMemoryProperties,
};

pub const PhysicalDeviceExternalBufferInfo = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
    pNext: ?*const c_void = null,
    flags: BufferCreateFlags,
    usage: BufferUsageFlags,
    handleType: ExternalMemoryHandleTypeFlags,
};

pub const ExternalBufferProperties = extern struct {
    sType: StructureType = .EXTERNAL_BUFFER_PROPERTIES,
    pNext: ?*c_void = null,
    externalMemoryProperties: ExternalMemoryProperties,
};

pub const PhysicalDeviceIDProperties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_ID_PROPERTIES,
    pNext: ?*c_void = null,
    deviceUUID: [16]u8,
    driverUUID: [16]u8,
    deviceLUID: [8]u8,
    deviceNodeMask: u32,
    deviceLUIDValid: Bool32,
};

pub const ExternalMemoryImageCreateInfo = extern struct {
    sType: StructureType = .EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    pNext: ?*const c_void = null,
    handleTypes: ExternalMemoryHandleTypeFlags,
};

pub const ExternalMemoryBufferCreateInfo = extern struct {
    sType: StructureType = .EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    pNext: ?*const c_void = null,
    handleTypes: ExternalMemoryHandleTypeFlags,
};

pub const ExportMemoryAllocateInfo = extern struct {
    sType: StructureType = .EXPORT_MEMORY_ALLOCATE_INFO,
    pNext: ?*const c_void = null,
    handleTypes: ExternalMemoryHandleTypeFlags,
};

pub const PhysicalDeviceExternalFenceInfo = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
    pNext: ?*const c_void = null,
    handleType: ExternalFenceHandleTypeFlags,
};

pub const ExternalFenceProperties = extern struct {
    sType: StructureType = .EXTERNAL_FENCE_PROPERTIES,
    pNext: ?*c_void = null,
    exportFromImportedHandleTypes: ExternalFenceHandleTypeFlags,
    compatibleHandleTypes: ExternalFenceHandleTypeFlags,
    externalFenceFeatures: ExternalFenceFeatureFlags,
};

pub const ExportFenceCreateInfo = extern struct {
    sType: StructureType = .EXPORT_FENCE_CREATE_INFO,
    pNext: ?*const c_void = null,
    handleTypes: ExternalFenceHandleTypeFlags,
};

pub const ExportSemaphoreCreateInfo = extern struct {
    sType: StructureType = .EXPORT_SEMAPHORE_CREATE_INFO,
    pNext: ?*const c_void = null,
    handleTypes: ExternalSemaphoreHandleTypeFlags,
};

pub const PhysicalDeviceExternalSemaphoreInfo = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
    pNext: ?*const c_void = null,
    handleType: ExternalSemaphoreHandleTypeFlags,
};

pub const ExternalSemaphoreProperties = extern struct {
    sType: StructureType = .EXTERNAL_SEMAPHORE_PROPERTIES,
    pNext: ?*c_void = null,
    exportFromImportedHandleTypes: ExternalSemaphoreHandleTypeFlags,
    compatibleHandleTypes: ExternalSemaphoreHandleTypeFlags,
    externalSemaphoreFeatures: ExternalSemaphoreFeatureFlags,
};

pub const PhysicalDeviceMaintenance3Properties = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
    pNext: ?*c_void = null,
    maxPerSetDescriptors: u32,
    maxMemoryAllocationSize: DeviceSize,
};

pub const DescriptorSetLayoutSupport = extern struct {
    sType: StructureType = .DESCRIPTOR_SET_LAYOUT_SUPPORT,
    pNext: ?*c_void = null,
    supported: Bool32,
};

pub const PhysicalDeviceShaderDrawParameterFeatures = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES,
    pNext: ?*c_void = null,
    shaderDrawParameters: Bool32,
};


pub extern fn vkEnumerateInstanceVersion(pApiVersion: *u32) Result;
pub extern fn vkBindBufferMemory2(device: Device, bindInfoCount: u32, pBindInfos: ?[*]const BindBufferMemoryInfo) Result;
pub extern fn vkBindImageMemory2(device: Device, bindInfoCount: u32, pBindInfos: ?[*]const BindImageMemoryInfo) Result;
pub extern fn vkGetDeviceGroupPeerMemoryFeatures(device: Device, heapIndex: u32, localDeviceIndex: u32, remoteDeviceIndex: u32, pPeerMemoryFeatures: *PeerMemoryFeatureFlags) void;
pub extern fn vkCmdSetDeviceMask(commandBuffer: CommandBuffer, deviceMask: u32) void;
pub extern fn vkCmdDispatchBase(commandBuffer: CommandBuffer, baseGroupX: u32, baseGroupY: u32, baseGroupZ: u32, groupCountX: u32, groupCountY: u32, groupCountZ: u32) void;
pub extern fn vkEnumeratePhysicalDeviceGroups(instance: Instance, pPhysicalDeviceGroupCount: *u32, pPhysicalDeviceGroupProperties: ?[*]PhysicalDeviceGroupProperties) Result;
pub extern fn vkGetImageMemoryRequirements2(device: Device, pInfo: *const ImageMemoryRequirementsInfo2, pMemoryRequirements: *MemoryRequirements2) void;
pub extern fn vkGetBufferMemoryRequirements2(device: Device, pInfo: *const BufferMemoryRequirementsInfo2, pMemoryRequirements: *MemoryRequirements2) void;
pub extern fn vkGetImageSparseMemoryRequirements2(device: Device, pInfo: *const ImageSparseMemoryRequirementsInfo2, pSparseMemoryRequirementCount: *u32, pSparseMemoryRequirements: ?[*]SparseImageMemoryRequirements2) void;
pub extern fn vkGetPhysicalDeviceFeatures2(physicalDevice: PhysicalDevice, pFeatures: *PhysicalDeviceFeatures2) void;
pub extern fn vkGetPhysicalDeviceProperties2(physicalDevice: PhysicalDevice, pProperties: *PhysicalDeviceProperties2) void;
pub extern fn vkGetPhysicalDeviceFormatProperties2(physicalDevice: PhysicalDevice, format: Format, pFormatProperties: *FormatProperties2) void;
pub extern fn vkGetPhysicalDeviceImageFormatProperties2(physicalDevice: PhysicalDevice, pImageFormatInfo: *const PhysicalDeviceImageFormatInfo2, pImageFormatProperties: *ImageFormatProperties2) Result;
pub extern fn vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice: PhysicalDevice, pQueueFamilyPropertyCount: *u32, pQueueFamilyProperties: ?[*]QueueFamilyProperties2) void;
pub extern fn vkGetPhysicalDeviceMemoryProperties2(physicalDevice: PhysicalDevice, pMemoryProperties: *PhysicalDeviceMemoryProperties2) void;
pub extern fn vkGetPhysicalDeviceSparseImageFormatProperties2(physicalDevice: PhysicalDevice, pFormatInfo: *const PhysicalDeviceSparseImageFormatInfo2, pPropertyCount: *u32, pProperties: ?[*]SparseImageFormatProperties2) void;
pub extern fn vkTrimCommandPool(device: Device, commandPool: CommandPool, flags: CommandPoolTrimFlags) void;
pub extern fn vkGetDeviceQueue2(device: Device, pQueueInfo: *const DeviceQueueInfo2, pQueue: *Queue) void;
pub extern fn vkCreateSamplerYcbcrConversion(device: Device, pCreateInfo: *const SamplerYcbcrConversionCreateInfo, pAllocator: ?*const AllocationCallbacks, pYcbcrConversion: *SamplerYcbcrConversion) Result;
pub extern fn vkDestroySamplerYcbcrConversion(device: Device, ycbcrConversion: SamplerYcbcrConversion, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkCreateDescriptorUpdateTemplate(device: Device, pCreateInfo: *const DescriptorUpdateTemplateCreateInfo, pAllocator: ?*const AllocationCallbacks, pDescriptorUpdateTemplate: *DescriptorUpdateTemplate) Result;
pub extern fn vkDestroyDescriptorUpdateTemplate(device: Device, descriptorUpdateTemplate: DescriptorUpdateTemplate, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkUpdateDescriptorSetWithTemplate(device: Device, descriptorSet: DescriptorSet, descriptorUpdateTemplate: DescriptorUpdateTemplate, pData: ?*const c_void) void;
pub extern fn vkGetPhysicalDeviceExternalBufferProperties(physicalDevice: PhysicalDevice, pExternalBufferInfo: *const PhysicalDeviceExternalBufferInfo, pExternalBufferProperties: *ExternalBufferProperties) void;
pub extern fn vkGetPhysicalDeviceExternalFenceProperties(physicalDevice: PhysicalDevice, pExternalFenceInfo: *const PhysicalDeviceExternalFenceInfo, pExternalFenceProperties: *ExternalFenceProperties) void;
pub extern fn vkGetPhysicalDeviceExternalSemaphoreProperties(physicalDevice: PhysicalDevice, pExternalSemaphoreInfo: *const PhysicalDeviceExternalSemaphoreInfo, pExternalSemaphoreProperties: *ExternalSemaphoreProperties) void;
pub extern fn vkGetDescriptorSetLayoutSupport(device: Device, pCreateInfo: *const DescriptorSetLayoutCreateInfo, pSupport: *DescriptorSetLayoutSupport) void;
