usingnamespace @import("vk_core.zig");


// ---------------------------------- VkSurfaceKHR ------------------------------------------

pub const struct_VkSurfaceKHR_T = @OpaqueType();
pub const SurfaceKHR = *struct_VkSurfaceKHR_T;

pub const ColorSpaceKHR = extern enum {
    SRGB_NONLINEAR_KHR = 0,
    DISPLAY_P3_NONLINEAR_EXT = 1000104001,
    EXTENDED_SRGB_LINEAR_EXT = 1000104002,
    DCI_P3_LINEAR_EXT = 1000104003,
    DCI_P3_NONLINEAR_EXT = 1000104004,
    BT709_LINEAR_EXT = 1000104005,
    BT709_NONLINEAR_EXT = 1000104006,
    BT2020_LINEAR_EXT = 1000104007,
    HDR10_ST2084_EXT = 1000104008,
    DOLBYVISION_EXT = 1000104009,
    HDR10_HLG_EXT = 1000104010,
    ADOBERGB_LINEAR_EXT = 1000104011,
    ADOBERGB_NONLINEAR_EXT = 1000104012,
    PASS_THROUGH_EXT = 1000104013,
    EXTENDED_SRGB_NONLINEAR_EXT = 1000104014,
};

pub const PresentModeKHR = extern enum {
    IMMEDIATE_KHR = 0,
    MAILBOX_KHR = 1,
    FIFO_KHR = 2,
    FIFO_RELAXED_KHR = 3,
    SHARED_DEMAND_REFRESH_KHR = 1000111000,
    SHARED_CONTINUOUS_REFRESH_KHR = 1000111001,
};


pub const SurfaceTransformFlagsKHR = Flags;
pub const SurfaceTransformFlagBitsKHR = struct {
    pub const IDENTITY_BIT_KHR = 1;
    pub const ROTATE_90_BIT_KHR = 2;
    pub const ROTATE_180_BIT_KHR = 4;
    pub const ROTATE_270_BIT_KHR = 8;
    pub const HORIZONTAL_MIRROR_BIT_KHR = 16;
    pub const HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR = 32;
    pub const HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR = 64;
    pub const HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR = 128;
    pub const INHERIT_BIT_KHR = 256;
};

pub const CompositeAlphaFlagsKHR = Flags;
pub const CompositeAlphaFlagBitsKHR = struct {
    pub const OPAQUE_BIT_KHR = 1;
    pub const PRE_MULTIPLIED_BIT_KHR = 2;
    pub const POST_MULTIPLIED_BIT_KHR = 4;
    pub const INHERIT_BIT_KHR = 8;
};


pub const SurfaceCapabilitiesKHR = extern struct {
    minImageCount: u32,
    maxImageCount: u32,
    currentExtent: Extent2D,
    minImageExtent: Extent2D,
    maxImageExtent: Extent2D,
    maxImageArrayLayers: u32,
    supportedTransforms: SurfaceTransformFlagsKHR,
    currentTransform: SurfaceTransformFlagsKHR,
    supportedCompositeAlpha: CompositeAlphaFlagsKHR,
    supportedUsageFlags: ImageUsageFlags,
};

pub const SurfaceFormatKHR = extern struct {
    format: Format,
    colorSpace: ColorSpaceKHR,
};

pub extern fn vkDestroySurfaceKHR(instance: Instance, surface: SurfaceKHR, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice: PhysicalDevice, queueFamilyIndex: u32, surface: SurfaceKHR, pSupported: *Bool32) Result;
pub extern fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice: PhysicalDevice, surface: SurfaceKHR, pSurfaceCapabilities: *SurfaceCapabilitiesKHR) Result;
pub extern fn vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice: PhysicalDevice, surface: SurfaceKHR, pSurfaceFormatCount: *u32, pSurfaceFormats: ?[*]SurfaceFormatKHR) Result;
pub extern fn vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice: PhysicalDevice, surface: SurfaceKHR, pPresentModeCount: *u32, pPresentModes: ?[*]PresentModeKHR) Result;


// -------------------------------------- VkSwapchainKHR ---------------------------------------------------

pub const struct_VkSwapchainKHR_T = @OpaqueType();
pub const SwapchainKHR = ?*struct_VkSwapchainKHR_T;

pub const SwapchainCreateFlagsKHR = Flags;
pub const SwapchainCreateFlagBitsKHR = struct {
    pub const SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR = 1;
    pub const PROTECTED_BIT_KHR = 2;
};

pub const DeviceGroupPresentModeFlagsKHR = Flags;
pub const DeviceGroupPresentModeFlagBitsKHR = struct {
    pub const LOCAL_BIT_KHR = 1,
    pub const REMOTE_BIT_KHR = 2,
    pub const SUM_BIT_KHR = 4,
    pub const LOCAL_MULTI_DEVICE_BIT_KHR = 8,
};


pub const SwapchainCreateInfoKHR = extern struct {
    sType: StructureType = .SWAPCHAIN_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    flags: SwapchainCreateFlagsKHR,
    surface: SurfaceKHR,
    minImageCount: u32,
    imageFormat: Format,
    imageColorSpace: ColorSpaceKHR,
    imageExtent: Extent2D,
    imageArrayLayers: u32,
    imageUsage: ImageUsageFlags,
    imageSharingMode: SharingMode,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: ?[*]const u32,
    preTransform: SurfaceTransformFlagsKHR,
    compositeAlpha: CompositeAlphaFlagsKHR,
    presentMode: PresentModeKHR,
    clipped: Bool32,
    oldSwapchain: SwapchainKHR,
};

pub const PresentInfoKHR = extern struct {
    sType: StructureType = .PRESENT_INFO_KHR,
    pNext: ?*const c_void = null,
    waitSemaphoreCount: u32,
    pWaitSemaphores: ?[*]const Semaphore,
    swapchainCount: u32,
    pSwapchains: ?[*]const SwapchainKHR,
    pImageIndices: ?[*]const u32,
    pResults: ?[*]Result,
};

pub const ImageSwapchainCreateInfoKHR = extern struct {
    sType: StructureType = .IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    swapchain: SwapchainKHR,
};

pub const BindImageMemorySwapchainInfoKHR = extern struct {
    sType: StructureType = .BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR,
    pNext: ?*const c_void = null,
    swapchain: SwapchainKHR,
    imageIndex: u32,
};

pub const AcquireNextImageInfoKHR = extern struct {
    sType: StructureType = .ACQUIRE_NEXT_IMAGE_INFO_KHR,
    pNext: ?*const c_void = null,
    swapchain: SwapchainKHR,
    timeout: u64,
    semaphore: Semaphore,
    fence: Fence,
    deviceMask: u32,
};

pub const DeviceGroupPresentCapabilitiesKHR = extern struct {
    sType: StructureType = .DEVICE_GROUP_PRESENT_CAPABILITIES_KHR,
    pNext: ?*const c_void = null,
    presentMask: [32]u32,
    modes: DeviceGroupPresentModeFlagsKHR,
};

pub const DeviceGroupPresentInfoKHR = extern struct {
    sType: StructureType = .DEVICE_GROUP_PRESENT_INFO_KHR,
    pNext: ?*const c_void = null,
    swapchainCount: u32,
    pDeviceMasks: ?[*]const u32,
    mode: DeviceGroupPresentModeFlagsKHR,
};

pub const DeviceGroupSwapchainCreateInfoKHR = extern struct {
    sType: StructureType = .DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    modes: DeviceGroupPresentModeFlagsKHR,
};


pub extern fn vkCreateSwapchainKHR(device: Device, pCreateInfo: *const SwapchainCreateInfoKHR, pAllocator: ?*const AllocationCallbacks, pSwapchain: *SwapchainKHR) Result;
pub extern fn vkDestroySwapchainKHR(device: Device, swapchain: SwapchainKHR, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkGetSwapchainImagesKHR(device: Device, swapchain: SwapchainKHR, pSwapchainImageCount: *u32, pSwapchainImages: ?[*]Image) Result;
pub extern fn vkAcquireNextImageKHR(device: Device, swapchain: SwapchainKHR, timeout: u64, semaphore: Semaphore, fence: Fence, pImageIndex: *u32) Result;
pub extern fn vkQueuePresentKHR(queue: Queue, pPresentInfo: *const PresentInfoKHR) Result;
pub extern fn vkGetDeviceGroupPresentCapabilitiesKHR(device: Device, pDeviceGroupPresentCapabilities: *DeviceGroupPresentCapabilitiesKHR) Result;
pub extern fn vkGetDeviceGroupSurfacePresentModesKHR(device: Device, surface: SurfaceKHR, pModes: *DeviceGroupPresentModeFlagsKHR) Result;
pub extern fn vkGetPhysicalDevicePresentRectanglesKHR(physicalDevice: PhysicalDevice, surface: SurfaceKHR, pRectCount: *u32, pRects: ?[*]Rect2D) Result;
pub extern fn vkAcquireNextImage2KHR(device: Device, pAcquireInfo: *const AcquireNextImageInfoKHR, pImageIndex: *u32) Result;


// ----------------------------------------- VkDisplayKHR ------------------------------------------

pub const struct_VkDisplayKHR_T = @OpaqueType();
pub const VkDisplayKHR = ?*struct_VkDisplayKHR_T;
pub const struct_VkDisplayModeKHR_T = @OpaqueType();
pub const VkDisplayModeKHR = ?*struct_VkDisplayModeKHR_T;


pub const DisplayPlaneAlphaFlagsKHR = Flags;
pub const DisplayPlaneAlphaFlagBitsKHR = struct {
    pub const OPAQUE_BIT_KHR = 1;
    pub const GLOBAL_BIT_KHR = 2;
    pub const PER_PIXEL_BIT_KHR = 4;
    pub const PER_PIXEL_PREMULTIPLIED_BIT_KHR = 8;
};

pub const DisplayModeCreateFlagsKHR = Flags;
pub const DisplaySurfaceCreateFlagsKHR = Flags;


pub const DisplayPropertiesKHR = extern struct {
    display: DisplayKHR,
    displayName: ?CString,
    physicalDimensions: Extent2D,
    physicalResolution: Extent2D,
    supportedTransforms: SurfaceTransformFlagsKHR,
    planeReorderPossible: Bool32,
    persistentContent: Bool32,
};

pub const DisplayModeParametersKHR = extern struct {
    visibleRegion: Extent2D,
    refreshRate: u32,
};

pub const DisplayModePropertiesKHR = extern struct {
    displayMode: DisplayModeKHR,
    parameters: DisplayModeParametersKHR,
};

pub const DisplayModeCreateInfoKHR = extern struct {
    sType: StructureType = .DISPLAY_MODE_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    flags: DisplayModeCreateFlagsKHR = 0,
    parameters: DisplayModeParametersKHR,
};

pub const DisplayPlaneCapabilitiesKHR = extern struct {
    supportedAlpha: DisplayPlaneAlphaFlagsKHR,
    minSrcPosition: Offset2D,
    maxSrcPosition: Offset2D,
    minSrcExtent: Extent2D,
    maxSrcExtent: Extent2D,
    minDstPosition: Offset2D,
    maxDstPosition: Offset2D,
    minDstExtent: Extent2D,
    maxDstExtent: Extent2D,
};

pub const DisplayPlanePropertiesKHR = extern struct {
    currentDisplay: DisplayKHR,
    currentStackIndex: u32,
};

pub const DisplaySurfaceCreateInfoKHR = extern struct {
    sType: StructureType = .DISPLAY_SURFACE_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    flags: DisplaySurfaceCreateFlagsKHR = 0,
    displayMode: DisplayModeKHR,
    planeIndex: u32,
    planeStackIndex: u32,
    transform: SurfaceTransformFlagsKHR,
    globalAlpha: f32,
    alphaMode: DisplayPlaneAlphaFlagsKHR,
    imageExtent: Extent2D,
};

pub const DisplayPresentInfoKHR = extern struct {
    sType: StructureType = .DISPLAY_PRESENT_INFO_KHR,
    pNext: ?*const c_void = null,
    srcRect: Rect2D,
    dstRect: Rect2D,
    persistent: Bool32,
};


pub extern fn vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice: PhysicalDevice, pPropertyCount: *u32, pProperties: ?[*]DisplayPropertiesKHR) Result;
pub extern fn vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice: PhysicalDevice, pPropertyCount: *u32, pProperties: ?[*]DisplayPlanePropertiesKHR) Result;
pub extern fn vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice: PhysicalDevice, planeIndex: u32, pDisplayCount: *u32, pDisplays: ?[*]DisplayKHR) Result;
pub extern fn vkGetDisplayModePropertiesKHR(physicalDevice: PhysicalDevice, display: DisplayKHR, pPropertyCount: *u32, pProperties: ?[*]DisplayModePropertiesKHR) Result;
pub extern fn vkCreateDisplayModeKHR(physicalDevice: PhysicalDevice, display: DisplayKHR, pCreateInfo: *const DisplayModeCreateInfoKHR, pAllocator: ?*const AllocationCallbacks, pMode: *DisplayModeKHR) Result;
pub extern fn vkGetDisplayPlaneCapabilitiesKHR(physicalDevice: PhysicalDevice, mode: DisplayModeKHR, planeIndex: u32, pCapabilities: *DisplayPlaneCapabilitiesKHR) Result;
pub extern fn vkCreateDisplayPlaneSurfaceKHR(instance: Instance, pCreateInfo: *const DisplaySurfaceCreateInfoKHR, pAllocator: ?*const AllocationCallbacks, pSurface: *SurfaceKHR) Result;
pub extern fn vkCreateSharedSwapchainsKHR(device: Device, swapchainCount: u32, pCreateInfos: ?[*]const SwapchainCreateInfoKHR, pAllocator: ?*const AllocationCallbacks, pSwapchains: ?[*]SwapchainKHR) Result;
pub extern fn vkGetPhysicalDeviceFeatures2KHR(physicalDevice: PhysicalDevice, pFeatures: *PhysicalDeviceFeatures2) void;
pub extern fn vkGetPhysicalDeviceProperties2KHR(physicalDevice: PhysicalDevice, pProperties: *PhysicalDeviceProperties2) void;
pub extern fn vkGetPhysicalDeviceFormatProperties2KHR(physicalDevice: PhysicalDevice, format: Format, pFormatProperties: *FormatProperties2) void;
pub extern fn vkGetPhysicalDeviceImageFormatProperties2KHR(physicalDevice: PhysicalDevice, pImageFormatInfo: *const PhysicalDeviceImageFormatInfo2, pImageFormatProperties: *ImageFormatProperties2) Result;
pub extern fn vkGetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice: PhysicalDevice, pQueueFamilyPropertyCount: *u32, pQueueFamilyProperties: ?[*]QueueFamilyProperties2) void;
pub extern fn vkGetPhysicalDeviceMemoryProperties2KHR(physicalDevice: PhysicalDevice, pMemoryProperties: *PhysicalDeviceMemoryProperties2) void;
pub extern fn vkGetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice: PhysicalDevice, pFormatInfo: *const PhysicalDeviceSparseImageFormatInfo2, pPropertyCount: *u32, pProperties: ?[*]SparseImageFormatProperties2) void;
pub extern fn vkGetDeviceGroupPeerMemoryFeaturesKHR(device: Device, heapIndex: u32, localDeviceIndex: u32, remoteDeviceIndex: u32, pPeerMemoryFeatures: *PeerMemoryFeatureFlags) void;
pub extern fn vkCmdSetDeviceMaskKHR(commandBuffer: CommandBuffer, deviceMask: u32) void;
pub extern fn vkCmdDispatchBaseKHR(commandBuffer: CommandBuffer, baseGroupX: u32, baseGroupY: u32, baseGroupZ: u32, groupCountX: u32, groupCountY: u32, groupCountZ: u32) void;
pub extern fn vkTrimCommandPoolKHR(device: Device, commandPool: CommandPool, flags: CommandPoolTrimFlags) void;
pub extern fn vkEnumeratePhysicalDeviceGroupsKHR(instance: Instance, pPhysicalDeviceGroupCount: *u32, pPhysicalDeviceGroupProperties: ?[*]PhysicalDeviceGroupProperties) Result;
pub extern fn vkGetPhysicalDeviceExternalBufferPropertiesKHR(physicalDevice: PhysicalDevice, pExternalBufferInfo: *const PhysicalDeviceExternalBufferInfo, pExternalBufferProperties: *ExternalBufferProperties) void;


// --------------------------------------- Other KHR Extensions ------------------------------------------------

pub const ImportMemoryFdInfoKHR = extern struct {
    sType: StructureType = .IMPORT_MEMORY_FD_INFO_KHR,
    pNext: ?*const c_void = null,
    handleType: ExternalMemoryHandleTypeFlags,
    fd: c_int,
};

pub const MemoryFdPropertiesKHR = extern struct {
    sType: StructureType = .MEMORY_FD_PROPERTIES_KHR,
    pNext: ?*c_void = null,
    memoryTypeBits: u32,
};

pub const MemoryGetFdInfoKHR = extern struct {
    sType: StructureType = .MEMORY_GET_FD_INFO_KHR,
    pNext: ?*const c_void = null,
    memory: DeviceMemory,
    handleType: ExternalMemoryHandleTypeFlags,
};

pub extern fn vkGetMemoryFdKHR(device: Device, pGetFdInfo: *const MemoryGetFdInfoKHR, pFd: *c_int) Result;
pub extern fn vkGetMemoryFdPropertiesKHR(device: Device, handleType: ExternalMemoryHandleTypeFlags, fd: c_int, pMemoryFdProperties: *MemoryFdPropertiesKHR) Result;
pub extern fn vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(physicalDevice: PhysicalDevice, pExternalSemaphoreInfo: *const PhysicalDeviceExternalSemaphoreInfo, pExternalSemaphoreProperties: *ExternalSemaphoreProperties) void;



pub const ImportSemaphoreFdInfoKHR = extern struct {
    sType: StructureType = .IMPORT_SEMAPHORE_FD_INFO_KHR,
    pNext: ?*const c_void = null,
    semaphore: Semaphore,
    flags: SemaphoreImportFlags,
    handleType: ExternalSemaphoreHandleTypeFlags,
    fd: c_int,
};

pub const SemaphoreGetFdInfoKHR = extern struct {
    sType: StructureType = .SEMAPHORE_GET_FD_INFO_KHR,
    pNext: ?*const c_void = null,
    semaphore: Semaphore,
    handleType: ExternalSemaphoreHandleTypeFlags,
};

pub extern fn vkImportSemaphoreFdKHR(device: Device, pImportSemaphoreFdInfo: *const ImportSemaphoreFdInfoKHR) Result;
pub extern fn vkGetSemaphoreFdKHR(device: Device, pGetFdInfo: *const SemaphoreGetFdInfoKHR, pFd: *c_int) Result;

pub const PhysicalDevicePushDescriptorPropertiesKHR = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
    pNext: ?*c_void = null,
    maxPushDescriptors: u32,
};

pub extern fn vkCmdPushDescriptorSetKHR(commandBuffer: CommandBuffer, pipelineBindPoint: PipelineBindPoint, layout: PipelineLayout, set: u32, descriptorWriteCount: u32, pDescriptorWrites: ?[*]const WriteDescriptorSet) void;
pub extern fn vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer: CommandBuffer, descriptorUpdateTemplate: DescriptorUpdateTemplate, layout: PipelineLayout, set: u32, pData: ?*const c_void) void;


pub const RectLayerKHR = extern struct {
    offset: Offset2D,
    extent: Extent2D,
    layer: u32,
};

pub const PresentRegionKHR = extern struct {
    rectangleCount: u32,
    pRectangles: ?[*]const RectLayerKHR,
};

pub const PresentRegionsKHR = extern struct {
    sType: StructureType = .PRESENT_REGIONS_KHR,
    pNext: ?*const c_void = null,
    swapchainCount: u32,
    pRegions: ?[*]const PresentRegionKHR,
};
pub extern fn vkCreateDescriptorUpdateTemplateKHR(device: Device, pCreateInfo: *const DescriptorUpdateTemplateCreateInfo, pAllocator: ?*const AllocationCallbacks, pDescriptorUpdateTemplate: *DescriptorUpdateTemplate) Result;
pub extern fn vkDestroyDescriptorUpdateTemplateKHR(device: Device, descriptorUpdateTemplate: DescriptorUpdateTemplate, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkUpdateDescriptorSetWithTemplateKHR(device: Device, descriptorSet: DescriptorSet, descriptorUpdateTemplate: DescriptorUpdateTemplate, pData: ?*const c_void) void;


pub const SharedPresentSurfaceCapabilitiesKHR = extern struct {
    sType: StructureType = .SHARED_PRESENT_SURFACE_CAPABILITIES_KHR,
    pNext: ?*c_void = null,
    sharedPresentSupportedUsageFlags: ImageUsageFlags,
};
pub extern fn vkGetSwapchainStatusKHR(device: Device, swapchain: SwapchainKHR) Result;
pub extern fn vkGetPhysicalDeviceExternalFencePropertiesKHR(physicalDevice: PhysicalDevice, pExternalFenceInfo: *const PhysicalDeviceExternalFenceInfo, pExternalFenceProperties: *ExternalFenceProperties) void;


pub const ImportFenceFdInfoKHR = extern struct {
    sType: StructureType = .IMPORT_FENCE_FD_INFO_KHR,
    pNext: ?*const c_void = null,
    fence: Fence,
    flags: FenceImportFlags,
    handleType: ExternalFenceHandleTypeFlags,
    fd: c_int,
};

pub const FenceGetFdInfoKHR = extern struct {
    sType: StructureType,
    pNext: ?*const c_void,
    fence: Fence,
    handleType: ExternalFenceHandleTypeFlags,
};

pub extern fn vkImportFenceFdKHR(device: Device, pImportFenceFdInfo: *const ImportFenceFdInfoKHR) Result;
pub extern fn vkGetFenceFdKHR(device: Device, pGetFdInfo: *const FenceGetFdInfoKHR, pFd: *c_int) Result;


pub const PhysicalDeviceSurfaceInfo2KHR = extern struct {
    sType: StructureType = .PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
    pNext: ?*const c_void = null,
    surface: SurfaceKHR,
};

pub const SurfaceCapabilities2KHR = extern struct {
    sType: StructureType = .SURFACE_CAPABILITIES_2_KHR,
    pNext: ?*c_void = null,
    surfaceCapabilities: SurfaceCapabilitiesKHR,
};

pub const SurfaceFormat2KHR = extern struct {
    sType: StructureType = .SURFACE_FORMAT_2_KHR,
    pNext: ?*c_void = null,
    surfaceFormat: SurfaceFormatKHR,
};

pub extern fn vkGetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice: PhysicalDevice, pSurfaceInfo: *const PhysicalDeviceSurfaceInfo2KHR, pSurfaceCapabilities: *SurfaceCapabilities2KHR) Result;
pub extern fn vkGetPhysicalDeviceSurfaceFormats2KHR(physicalDevice: PhysicalDevice, pSurfaceInfo: *const PhysicalDeviceSurfaceInfo2KHR, pSurfaceFormatCount: *u32, pSurfaceFormats: ?[*]SurfaceFormat2KHR) Result;


pub const DisplayProperties2KHR = extern struct {
    sType: StructureType = .DISPLAY_PROPERTIES_2_KHR,
    pNext: ?*c_void = null,
    displayProperties: DisplayPropertiesKHR,
};

pub const DisplayPlaneProperties2KHR = extern struct {
    sType: StructureType = .DISPLAY_PLANE_PROPERTIES_2_KHR,
    pNext: ?*c_void = null,
    displayPlaneProperties: DisplayPlanePropertiesKHR,
};

pub const DisplayModeProperties2KHR = extern struct {
    sType: StructureType = .DISPLAY_MODE_PROPERTIES_2_KHR,
    pNext: ?*c_void = null,
    displayModeProperties: DisplayModePropertiesKHR,
};

pub const DisplayPlaneInfo2KHR = extern struct {
    sType: StructureType = .DISPLAY_PLANE_INFO_2_KHR,
    pNext: ?*const c_void = null,
    mode: DisplayModeKHR,
    planeIndex: u32,
};

pub const DisplayPlaneCapabilities2KHR = extern struct {
    sType: StructureType = .DISPLAY_PLANE_CAPABILITIES_2_KHR,
    pNext: ?*c_void = null,
    capabilities: DisplayPlaneCapabilitiesKHR,
};

pub extern fn vkGetPhysicalDeviceDisplayProperties2KHR(physicalDevice: PhysicalDevice, pPropertyCount: *u32, pProperties: ?[*]DisplayProperties2KHR) Result;
pub extern fn vkGetPhysicalDeviceDisplayPlaneProperties2KHR(physicalDevice: PhysicalDevice, pPropertyCount: *u32, pProperties: ?[*]DisplayPlaneProperties2KHR) Result;
pub extern fn vkGetDisplayModeProperties2KHR(physicalDevice: PhysicalDevice, display: DisplayKHR, pPropertyCount: *u32, pProperties: ?[*]DisplayModeProperties2KHR) Result;
pub extern fn vkGetDisplayPlaneCapabilities2KHR(physicalDevice: PhysicalDevice, pDisplayPlaneInfo: *const DisplayPlaneInfo2KHR, pCapabilities: *DisplayPlaneCapabilities2KHR) Result;
pub extern fn vkGetImageMemoryRequirements2KHR(device: Device, pInfo: *const ImageMemoryRequirementsInfo2, pMemoryRequirements: *MemoryRequirements2) void;
pub extern fn vkGetBufferMemoryRequirements2KHR(device: Device, pInfo: *const BufferMemoryRequirementsInfo2, pMemoryRequirements: *MemoryRequirements2) void;
pub extern fn vkGetImageSparseMemoryRequirements2KHR(device: Device, pInfo: *const ImageSparseMemoryRequirementsInfo2, pSparseMemoryRequirementCount: *u32, pSparseMemoryRequirements: ?[*]SparseImageMemoryRequirements2) void;


pub const ImageFormatListCreateInfoKHR = extern struct {
    sType: StructureType = .IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
    pNext: ?*const c_void = null,
    viewFormatCount: u32,
    pViewFormats: ?[*]const Format,
};

pub extern fn vkCreateSamplerYcbcrConversionKHR(device: Device, pCreateInfo: *const SamplerYcbcrConversionCreateInfo, pAllocator: ?*const AllocationCallbacks, pYcbcrConversion: *SamplerYcbcrConversion) Result;
pub extern fn vkDestroySamplerYcbcrConversionKHR(device: Device, ycbcrConversion: SamplerYcbcrConversion, pAllocator: ?*const AllocationCallbacks) void;
pub extern fn vkBindBufferMemory2KHR(device: Device, bindInfoCount: u32, pBindInfos: ?[*]const BindBufferMemoryInfo) Result;
pub extern fn vkBindImageMemory2KHR(device: Device, bindInfoCount: u32, pBindInfos: ?[*]const BindImageMemoryInfo) Result;
pub extern fn vkGetDescriptorSetLayoutSupportKHR(device: Device, pCreateInfo: *const DescriptorSetLayoutCreateInfo, pSupport: *DescriptorSetLayoutSupport) void;
pub extern fn vkCmdDrawIndirectCountKHR(commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize, countBuffer: Buffer, countBufferOffset: DeviceSize, maxDrawCount: u32, stride: u32) void;
pub extern fn vkCmdDrawIndexedIndirectCountKHR(commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize, countBuffer: Buffer, countBufferOffset: DeviceSize, maxDrawCount: u32, stride: u32) void;


// ------------------------------------------- Backwards-compatibility Definitions -----------------------------

pub const RenderPassMultiviewCreateInfoKHR = RenderPassMultiviewCreateInfo;
pub const PhysicalDeviceMultiviewFeaturesKHR = PhysicalDeviceMultiviewFeatures;
pub const PhysicalDeviceMultiviewPropertiesKHR = PhysicalDeviceMultiviewProperties;
pub const PhysicalDeviceFeatures2KHR = PhysicalDeviceFeatures2;
pub const PhysicalDeviceProperties2KHR = PhysicalDeviceProperties2;
pub const FormatProperties2KHR = FormatProperties2;
pub const ImageFormatProperties2KHR = ImageFormatProperties2;
pub const PhysicalDeviceImageFormatInfo2KHR = PhysicalDeviceImageFormatInfo2;
pub const QueueFamilyProperties2KHR = QueueFamilyProperties2;
pub const PhysicalDeviceMemoryProperties2KHR = PhysicalDeviceMemoryProperties2;
pub const SparseImageFormatProperties2KHR = SparseImageFormatProperties2;
pub const PhysicalDeviceSparseImageFormatInfo2KHR = PhysicalDeviceSparseImageFormatInfo2;
pub const PeerMemoryFeatureFlagsKHR = PeerMemoryFeatureFlags;
pub const PeerMemoryFeatureFlagBitsKHR = PeerMemoryFeatureFlagBits;
pub const MemoryAllocateFlagsKHR = MemoryAllocateFlags;
pub const MemoryAllocateFlagBitsKHR = MemoryAllocateFlagBits;
pub const MemoryAllocateFlagsInfoKHR = MemoryAllocateFlagsInfo;
pub const DeviceGroupRenderPassBeginInfoKHR = DeviceGroupRenderPassBeginInfo;
pub const DeviceGroupCommandBufferBeginInfoKHR = DeviceGroupCommandBufferBeginInfo;
pub const DeviceGroupSubmitInfoKHR = DeviceGroupSubmitInfo;
pub const DeviceGroupBindSparseInfoKHR = DeviceGroupBindSparseInfo;
pub const BindBufferMemoryDeviceGroupInfoKHR = BindBufferMemoryDeviceGroupInfo;
pub const BindImageMemoryDeviceGroupInfoKHR = BindImageMemoryDeviceGroupInfo;
pub const CommandPoolTrimFlagsKHR = CommandPoolTrimFlags;
pub const PhysicalDeviceGroupPropertiesKHR = PhysicalDeviceGroupProperties;
pub const DeviceGroupDeviceCreateInfoKHR = DeviceGroupDeviceCreateInfo;
pub const ExternalMemoryHandleTypeFlagsKHR = ExternalMemoryHandleTypeFlags;
pub const ExternalMemoryHandleTypeFlagBitsKHR = ExternalMemoryHandleTypeFlagBits;
pub const ExternalMemoryFeatureFlagsKHR = ExternalMemoryFeatureFlags;
pub const ExternalMemoryFeatureFlagBitsKHR = ExternalMemoryFeatureFlagBits;
pub const ExternalMemoryPropertiesKHR = ExternalMemoryProperties;
pub const PhysicalDeviceExternalImageFormatInfoKHR = PhysicalDeviceExternalImageFormatInfo;
pub const ExternalImageFormatPropertiesKHR = ExternalImageFormatProperties;
pub const PhysicalDeviceExternalBufferInfoKHR = PhysicalDeviceExternalBufferInfo;
pub const ExternalBufferPropertiesKHR = ExternalBufferProperties;
pub const PhysicalDeviceIDPropertiesKHR = PhysicalDeviceIDProperties;
pub const ExternalMemoryImageCreateInfoKHR = ExternalMemoryImageCreateInfo;
pub const ExternalMemoryBufferCreateInfoKHR = ExternalMemoryBufferCreateInfo;
pub const ExportMemoryAllocateInfoKHR = ExportMemoryAllocateInfo;
pub const ExternalSemaphoreHandleTypeFlagsKHR = ExternalSemaphoreHandleTypeFlags;
pub const ExternalSemaphoreHandleTypeFlagBitsKHR = ExternalSemaphoreHandleTypeFlagBits;
pub const ExternalSemaphoreFeatureFlagsKHR = ExternalSemaphoreFeatureFlags;
pub const ExternalSemaphoreFeatureFlagBitsKHR = ExternalSemaphoreFeatureFlagBits;
pub const PhysicalDeviceExternalSemaphoreInfoKHR = PhysicalDeviceExternalSemaphoreInfo;
pub const ExternalSemaphorePropertiesKHR = ExternalSemaphoreProperties;
pub const SemaphoreImportFlagsKHR = SemaphoreImportFlags;
pub const SemaphoreImportFlagBitsKHR = SemaphoreImportFlagBits;
pub const ExportSemaphoreCreateInfoKHR = ExportSemaphoreCreateInfo;
pub const PhysicalDevice16BitStorageFeaturesKHR = PhysicalDevice16BitStorageFeatures;
pub const DescriptorUpdateTemplateKHR = DescriptorUpdateTemplate;
pub const DescriptorUpdateTemplateTypeKHR = DescriptorUpdateTemplateType;
pub const DescriptorUpdateTemplateCreateFlagsKHR = DescriptorUpdateTemplateCreateFlags;
pub const DescriptorUpdateTemplateEntryKHR = DescriptorUpdateTemplateEntry;
pub const DescriptorUpdateTemplateCreateInfoKHR = DescriptorUpdateTemplateCreateInfo;
pub const ExternalFenceHandleTypeFlagsKHR = ExternalFenceHandleTypeFlags;
pub const ExternalFenceHandleTypeFlagBitsKHR = ExternalFenceHandleTypeFlagBits;
pub const ExternalFenceFeatureFlagsKHR = ExternalFenceFeatureFlags;
pub const ExternalFenceFeatureFlagBitsKHR = ExternalFenceFeatureFlagBits;
pub const PhysicalDeviceExternalFenceInfoKHR = PhysicalDeviceExternalFenceInfo;
pub const ExternalFencePropertiesKHR = ExternalFenceProperties;
pub const FenceImportFlagsKHR = FenceImportFlags;
pub const FenceImportFlagBitsKHR = FenceImportFlagBits;
pub const ExportFenceCreateInfoKHR = ExportFenceCreateInfo;
pub const PointClippingBehaviorKHR = PointClippingBehavior;
pub const TessellationDomainOriginKHR = TessellationDomainOrigin;
pub const PhysicalDevicePointClippingPropertiesKHR = PhysicalDevicePointClippingProperties;
pub const RenderPassInputAttachmentAspectCreateInfoKHR = RenderPassInputAttachmentAspectCreateInfo;
pub const InputAttachmentAspectReferenceKHR = InputAttachmentAspectReference;
pub const ImageViewUsageCreateInfoKHR = ImageViewUsageCreateInfo;
pub const PipelineTessellationDomainOriginStateCreateInfoKHR = PipelineTessellationDomainOriginStateCreateInfo;
pub const PhysicalDeviceVariablePointerFeaturesKHR = PhysicalDeviceVariablePointerFeatures;
pub const MemoryDedicatedRequirementsKHR = MemoryDedicatedRequirements;
pub const MemoryDedicatedAllocateInfoKHR = MemoryDedicatedAllocateInfo;
pub const BufferMemoryRequirementsInfo2KHR = BufferMemoryRequirementsInfo2;
pub const ImageMemoryRequirementsInfo2KHR = ImageMemoryRequirementsInfo2;
pub const ImageSparseMemoryRequirementsInfo2KHR = ImageSparseMemoryRequirementsInfo2;
pub const MemoryRequirements2KHR = MemoryRequirements2;
pub const SparseImageMemoryRequirements2KHR = SparseImageMemoryRequirements2;
pub const SamplerYcbcrConversionKHR = SamplerYcbcrConversion;
pub const SamplerYcbcrModelConversionKHR = SamplerYcbcrModelConversion;
pub const SamplerYcbcrRangeKHR = SamplerYcbcrRange;
pub const ChromaLocationKHR = ChromaLocation;
pub const SamplerYcbcrConversionCreateInfoKHR = SamplerYcbcrConversionCreateInfo;
pub const SamplerYcbcrConversionInfoKHR = SamplerYcbcrConversionInfo;
pub const BindImagePlaneMemoryInfoKHR = BindImagePlaneMemoryInfo;
pub const ImagePlaneMemoryRequirementsInfoKHR = ImagePlaneMemoryRequirementsInfo;
pub const PhysicalDeviceSamplerYcbcrConversionFeaturesKHR = PhysicalDeviceSamplerYcbcrConversionFeatures;
pub const SamplerYcbcrConversionImageFormatPropertiesKHR = SamplerYcbcrConversionImageFormatProperties;
pub const BindBufferMemoryInfoKHR = BindBufferMemoryInfo;
pub const BindImageMemoryInfoKHR = BindImageMemoryInfo;
pub const PhysicalDeviceMaintenance3PropertiesKHR = PhysicalDeviceMaintenance3Properties;
pub const DescriptorSetLayoutSupportKHR = DescriptorSetLayoutSupport;
