// @todo: big cleanup things:
// [x] Replace [*] with * in all applicable vk bindings
// [x] Make all vulkan bitfield enums into integers
// [x] Use structured types for buffer data
// [x] Set sType and pNext defaults for all vk structs

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const math = std.math;
const maxInt = math.maxInt;

const vk = @import("vk.zig");
const glfw = @import("glfw.zig");

const util = @import("util.zig");
const arrayPtr = util.arrayPtr;
const emptySlice = util.emptySlice;

const geo = @import("geo/geo.zig");
const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;
const Vec4 = geo.Vec4;
const Rotor3 = geo.Rotor3;
const Mat3 = geo.Mat3;
const Mat4 = geo.Mat4;
const colors = @import("color.zig");
const Color3f = colors.Color3f;

const WIDTH = 800;
const HEIGHT = 600;

const MAX_FRAMES_IN_FLIGHT = 2;

const enableValidationLayers = std.debug.runtime_safety;
const validationLayers = [_]vk.CString{"VK_LAYER_LUNARG_standard_validation"};
const deviceExtensions = [_]vk.CString{vk.KHR_SWAPCHAIN_EXTENSION_NAME};

const vertexData = [_]Vertex{
    Vertex.init(Vec2.init(0.5, -0.5), Color3f.init(1, 0.5, 0)),
    Vertex.init(Vec2.init(0.5, 0.5), Color3f.init(0, 1, 1)),
    Vertex.init(Vec2.init(-0.5, 0.5), Color3f.init(0.5, 0, 1)),
    Vertex.init(Vec2.init(-0.5, 0.5), Color3f.init(0.5, 0, 1)),
    Vertex.init(Vec2.init(-0.5, -0.5), Color3f.init(0, 1, 1)),
    Vertex.init(Vec2.init(0.5, -0.5), Color3f.init(1, 0.5, 0)),
};

var currentFrame: usize = 0;
var instance: vk.Instance = undefined;
var callback: vk.DebugReportCallbackEXT = undefined;
var surface: vk.SurfaceKHR = undefined;
var physicalDevice: vk.PhysicalDevice = undefined;
var global_device: vk.Device = undefined;
var graphicsQueue: vk.Queue = undefined;
var presentQueue: vk.Queue = undefined;
var swapChainImages: []vk.Image = undefined;
var swapChain: vk.SwapchainKHR = undefined;
var swapChainImageFormat: vk.Format = undefined;
var swapChainExtent: vk.Extent2D = undefined;
var swapChainImageViews: []vk.ImageView = undefined;
var renderPass: vk.RenderPass = undefined;
var descriptorSetLayouts: [1]vk.DescriptorSetLayout = undefined;
var descriptorPool: vk.DescriptorPool = undefined;
var descriptorSets: []vk.DescriptorSet = undefined;
var uniformBuffers: []vk.Buffer = undefined;
var uniformBuffersMemory: []vk.DeviceMemory = undefined;
var pipelineLayout: vk.PipelineLayout = undefined;
var graphicsPipeline: vk.Pipeline = undefined;
var swapChainFramebuffers: []vk.Framebuffer = undefined;
var commandPool: vk.CommandPool = undefined;
var commandBuffers: []vk.CommandBuffer = undefined;
var vertexBuffer: vk.Buffer = undefined;
var vertexBufferMemory: vk.DeviceMemory = undefined;

var imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
var renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
var inFlightFences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = undefined;

var startupTimeMillis: u64 = undefined;

fn currentTime() f32 {
    // TODO: This is not monotonic.  It will fail on leap years or other
    // cases where computer time stops or goes backwards.  It also can't
    // handle extremely long run durations and has trouble with hibernation.
    return @intToFloat(f32, std.time.milliTimestamp() - startupTimeMillis) * 0.001;
}

const Vertex = extern struct {
    pos: Vec2,
    color: Color3f,

    pub fn init(pos: Vec2, color: Color3f) Vertex {
        return .{
            .pos = pos,
            .color = color,
        };
    }

    const BindingDescriptions = [_]vk.VertexInputBindingDescription{.{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .inputRate = .VERTEX,
    }};

    const AttributeDescriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .R32G32_SFLOAT,
            .offset = @byteOffsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .R32G32B32_SFLOAT,
            .offset = @byteOffsetOf(Vertex, "color"),
        },
    };
};

const UniformBufferObject = extern struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
};

const QueueFamilyIndices = struct {
    graphicsFamily: ?u32,
    presentFamily: ?u32,

    fn init() QueueFamilyIndices {
        return .{
            .graphicsFamily = null,
            .presentFamily = null,
        };
    }

    fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphicsFamily != null and self.presentFamily != null;
    }
};

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR,
    formats: std.ArrayList(vk.SurfaceFormatKHR),
    presentModes: std.ArrayList(vk.PresentModeKHR),

    fn init(allocator: *Allocator) SwapChainSupportDetails {
        return .{
            .capabilities = mem.zeroes(vk.SurfaceCapabilitiesKHR),
            .formats = std.ArrayList(vk.SurfaceFormatKHR).init(allocator),
            .presentModes = std.ArrayList(vk.PresentModeKHR).init(allocator),
        };
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        self.formats.deinit();
        self.presentModes.deinit();
    }
};

pub fn main() !void {
    startupTimeMillis = std.time.milliTimestamp();

    if (glfw.glfwInit() == 0) return error.GlfwInitFailed;
    defer glfw.glfwTerminate();

    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_NO_API);
    glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, glfw.GLFW_FALSE);

    const window = glfw.glfwCreateWindow(WIDTH, HEIGHT, "Zig Vulkan Triangle", null, null) orelse return error.GlfwCreateWindowFailed;
    defer glfw.glfwDestroyWindow(window);

    const allocator = std.heap.c_allocator;
    try initVulkan(allocator, window);

    while (glfw.glfwWindowShouldClose(window) == 0) {
        glfw.glfwPollEvents();
        try drawFrame();
    }
    try vk.DeviceWaitIdle(global_device);

    cleanup();
}

fn cleanup() void {
    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        vk.DestroySemaphore(global_device, renderFinishedSemaphores[i], null);
        vk.DestroySemaphore(global_device, imageAvailableSemaphores[i], null);
        vk.DestroyFence(global_device, inFlightFences[i], null);
    }

    vk.DestroyCommandPool(global_device, commandPool, null);

    for (swapChainFramebuffers) |framebuffer| {
        vk.DestroyFramebuffer(global_device, framebuffer, null);
    }

    vk.DestroyPipeline(global_device, graphicsPipeline, null);
    vk.DestroyPipelineLayout(global_device, pipelineLayout, null);
    vk.DestroyRenderPass(global_device, renderPass, null);

    for (swapChainImageViews) |imageView| {
        vk.DestroyImageView(global_device, imageView, null);
    }

    vk.DestroySwapchainKHR(global_device, swapChain, null);

    for (uniformBuffers) |buffer| {
        vk.DestroyBuffer(global_device, buffer, null);
    }

    for (uniformBuffersMemory) |uniformMem| {
        vk.FreeMemory(global_device, uniformMem, null);
    }

    vk.DestroyDescriptorPool(global_device, descriptorPool, null);
    vk.DestroyDescriptorSetLayout(global_device, descriptorSetLayouts[0], null);

    vk.DestroyBuffer(global_device, vertexBuffer, null);
    vk.FreeMemory(global_device, vertexBufferMemory, null);
    vk.DestroyDevice(global_device, null);

    if (enableValidationLayers) {
        DestroyDebugReportCallbackEXT(null);
    }

    vk.DestroySurfaceKHR(instance, surface, null);
    vk.DestroyInstance(instance, null);
}

fn initVulkan(allocator: *Allocator, window: *glfw.GLFWwindow) !void {
    try createInstance(allocator);
    try setupDebugCallback();
    try createSurface(window);
    try pickPhysicalDevice(allocator);
    try createLogicalDevice(allocator);
    try createSwapChain(allocator);
    try createImageViews(allocator);
    try createRenderPass();
    try createDescriptorSetLayout();
    try createGraphicsPipeline(allocator);
    try createFramebuffers(allocator);
    try createCommandPool(allocator);
    try createVertexBuffer(allocator);
    try createUniformBuffers(allocator);
    try createDescriptorPool();
    try createDescriptorSets(allocator);
    try createCommandBuffers(allocator);
    try createSyncObjects();
}

fn findMemoryType(typeFilter: u32, properties: vk.MemoryPropertyFlags) !u32 {
    var memProperties = vk.GetPhysicalDeviceMemoryProperties(physicalDevice);

    var i: u32 = 0;
    while (i < memProperties.memoryTypeCount) : (i += 1) {
        if ((typeFilter & (@as(u32, 1) << @intCast(u5, i))) != 0 and memProperties.memoryTypes[i].propertyFlags.hasAllSet(properties))
            return i;
    }

    return error.NoSuitableMemory;
}

fn createBuffer(size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, outBuffer: *vk.Buffer, outBufferMemory: *vk.DeviceMemory) !void {
    const bufferInfo = vk.BufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharingMode = .EXCLUSIVE,
    };

    var buffer = try vk.CreateBuffer(global_device, bufferInfo, null);

    var memRequirements = vk.GetBufferMemoryRequirements(global_device, buffer);

    const allocInfo = vk.MemoryAllocateInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = try findMemoryType(memRequirements.memoryTypeBits, properties),
    };
    var bufferMemory = try vk.AllocateMemory(global_device, allocInfo, null);
    try vk.BindBufferMemory(global_device, buffer, bufferMemory, 0);

    outBuffer.* = buffer;
    outBufferMemory.* = bufferMemory;
}

fn copyBuffer(srcBuffer: vk.Buffer, dstBuffer: vk.Buffer, size: vk.DeviceSize) !void {
    const allocInfo = vk.CommandBufferAllocateInfo{
        .level = .PRIMARY,
        .commandPool = commandPool,
        .commandBufferCount = 1,
    };

    var commandBuffer: vk.CommandBuffer = undefined;
    try vk.AllocateCommandBuffers(global_device, allocInfo, arrayPtr(&commandBuffer));

    const beginInfo = vk.CommandBufferBeginInfo{
        .flags = .{ .oneTimeSubmit = true },
        .pInheritanceInfo = null,
    };
    try vk.BeginCommandBuffer(commandBuffer, beginInfo);

    const copyRegion = vk.BufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };
    vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, arrayPtr(&copyRegion));

    try vk.EndCommandBuffer(commandBuffer);

    const submitInfo = vk.SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = arrayPtr(&commandBuffer),
    };

    try vk.QueueSubmit(graphicsQueue, arrayPtr(&submitInfo), null);
    try vk.QueueWaitIdle(graphicsQueue);

    vk.FreeCommandBuffers(global_device, commandPool, arrayPtr(&commandBuffer));
}

fn createVertexBuffer(allocator: *Allocator) !void {
    const bufferSize: vk.DeviceSize = @sizeOf(@TypeOf(vertexData));

    var stagingBuffer: vk.Buffer = undefined;
    var stagingBufferMemory: vk.DeviceMemory = undefined;
    try createBuffer(
        bufferSize,
        .{ .transferSrc = true },
        .{ .hostVisible = true, .hostCoherent = true },
        &stagingBuffer,
        &stagingBufferMemory,
    );

    var data: *c_void = undefined;
    try vk.MapMemory(global_device, stagingBufferMemory, 0, bufferSize, .{}, &data);
    @memcpy(@ptrCast([*]u8, data), @ptrCast([*]const u8, &vertexData), bufferSize);
    vk.UnmapMemory(global_device, stagingBufferMemory);

    try createBuffer(
        bufferSize,
        .{ .transferDst = true, .vertexBuffer = true },
        .{ .deviceLocal = true },
        &vertexBuffer,
        &vertexBufferMemory,
    );

    try copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vk.DestroyBuffer(global_device, stagingBuffer, null);
    vk.FreeMemory(global_device, stagingBufferMemory, null);
}

fn createUniformBuffers(allocator: *Allocator) !void {
    const bufferSize = @as(vk.DeviceSize, @sizeOf(UniformBufferObject));

    uniformBuffers = try allocator.alloc(vk.Buffer, swapChainImages.len);
    uniformBuffersMemory = try allocator.alloc(vk.DeviceMemory, swapChainImages.len);

    for (uniformBuffers) |*buffer, i| {
        try createBuffer(
            bufferSize,
            .{ .uniformBuffer = true },
            .{ .hostVisible = true, .hostCoherent = true },
            buffer,
            &uniformBuffersMemory[i],
        );
    }
}

fn createDescriptorPool() !void {
    const poolSizes = [_]vk.DescriptorPoolSize{.{
        .inType = .UNIFORM_BUFFER,
        .descriptorCount = @intCast(u32, swapChainImages.len),
    }};

    const poolInfo = vk.DescriptorPoolCreateInfo{
        .poolSizeCount = poolSizes.len,
        .pPoolSizes = &poolSizes,
        .maxSets = @intCast(u32, swapChainImages.len),
    };

    descriptorPool = try vk.CreateDescriptorPool(global_device, poolInfo, null);
}

fn createDescriptorSets(allocator: *Allocator) !void {
    const layouts = try allocator.alloc(vk.DescriptorSetLayout, swapChainImages.len);
    defer allocator.free(layouts);

    for (layouts) |*layout| layout.* = descriptorSetLayouts[0];

    const allocInfo = vk.DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool,
        .descriptorSetCount = @intCast(u32, layouts.len),
        .pSetLayouts = layouts.ptr,
    };

    descriptorSets = try allocator.alloc(vk.DescriptorSet, swapChainImages.len);
    errdefer allocator.free(descriptorSets);

    try vk.AllocateDescriptorSets(global_device, allocInfo, descriptorSets);

    for (uniformBuffers) |buffer, i| {
        const bufferInfo = vk.DescriptorBufferInfo{
            .buffer = buffer,
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        const write = vk.WriteDescriptorSet{
            .dstSet = descriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = .UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = arrayPtr(&bufferInfo),
            .pImageInfo = undefined,
            .pTexelBufferView = undefined,
        };

        vk.UpdateDescriptorSets(global_device, arrayPtr(&write), emptySlice(vk.CopyDescriptorSet));
    }
}

fn createCommandBuffers(allocator: *Allocator) !void {
    commandBuffers = try allocator.alloc(vk.CommandBuffer, swapChainFramebuffers.len);

    const allocInfo = vk.CommandBufferAllocateInfo{
        .commandPool = commandPool,
        .level = .PRIMARY,
        .commandBufferCount = @intCast(u32, commandBuffers.len),
    };

    try vk.AllocateCommandBuffers(global_device, allocInfo, commandBuffers);

    for (commandBuffers) |command_buffer, i| {
        const beginInfo = vk.CommandBufferBeginInfo{
            .flags = .{ .simultaneousUse = true },
            .pInheritanceInfo = null,
        };

        try vk.BeginCommandBuffer(commandBuffers[i], beginInfo);

        const clearColor = vk.ClearValue{ .color = .{ .float32 = .{ 0.2, 0.2, 0.2, 1.0 } } };

        const renderPassInfo = vk.RenderPassBeginInfo{
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[i],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = arrayPtr(&clearColor),
        };

        vk.CmdBeginRenderPass(commandBuffers[i], renderPassInfo, .INLINE);
        {
            vk.CmdBindPipeline(commandBuffers[i], .GRAPHICS, graphicsPipeline);

            const offsets = [_]vk.DeviceSize{0};
            vk.CmdBindVertexBuffers(commandBuffers[i], 0, arrayPtr(&vertexBuffer), &offsets);

            vk.CmdBindDescriptorSets(commandBuffers[i], .GRAPHICS, pipelineLayout, 0, arrayPtr(&descriptorSets[i]), emptySlice(u32));

            vk.CmdDraw(commandBuffers[i], vertexData.len, 1, 0, 0);
        }
        vk.CmdEndRenderPass(commandBuffers[i]);

        try vk.EndCommandBuffer(commandBuffers[i]);
    }
}

fn createSyncObjects() !void {
    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        imageAvailableSemaphores[i] = try vk.CreateSemaphore(global_device, .{}, null);
        renderFinishedSemaphores[i] = try vk.CreateSemaphore(global_device, .{}, null);
        inFlightFences[i] = try vk.CreateFence(global_device, .{ .flags = .{ .signaled = true } }, null);
    }
}

fn createCommandPool(allocator: *Allocator) !void {
    const queueFamilyIndices = try findQueueFamilies(allocator, physicalDevice);

    const poolInfo = vk.CommandPoolCreateInfo{
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.?,
    };

    commandPool = try vk.CreateCommandPool(global_device, poolInfo, null);
}

fn createFramebuffers(allocator: *Allocator) !void {
    swapChainFramebuffers = try allocator.alloc(vk.Framebuffer, swapChainImageViews.len);

    for (swapChainImageViews) |swap_chain_image_view, i| {
        const framebufferInfo = vk.FramebufferCreateInfo{
            .renderPass = renderPass,
            .attachmentCount = 1,
            .pAttachments = arrayPtr(&swap_chain_image_view),
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1,
        };

        swapChainFramebuffers[i] = try vk.CreateFramebuffer(global_device, framebufferInfo, null);
    }
}

fn createShaderModule(code: []align(@alignOf(u32)) const u8) !vk.ShaderModule {
    const createInfo = vk.ShaderModuleCreateInfo{
        .codeSize = code.len,
        .pCode = @ptrCast([*]const u32, code.ptr),
    };

    return try vk.CreateShaderModule(global_device, createInfo, null);
}

fn createGraphicsPipeline(allocator: *Allocator) !void {
    const vertShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\vert.spv", ~@as(usize, 0), @alignOf(u32));
    defer allocator.free(vertShaderCode);

    const fragShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\frag.spv", ~@as(usize, 0), @alignOf(u32));
    defer allocator.free(fragShaderCode);

    const vertShaderModule = try createShaderModule(vertShaderCode);
    const fragShaderModule = try createShaderModule(fragShaderCode);

    const shaderStages = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex = true },
            .module = vertShaderModule,
            .pName = "main",
        },
        .{
            .stage = .{ .fragment = true },
            .module = fragShaderModule,
            .pName = "main",
        },
    };

    const vertexInputInfo = vk.PipelineVertexInputStateCreateInfo{
        .vertexBindingDescriptionCount = Vertex.BindingDescriptions.len,
        .vertexAttributeDescriptionCount = Vertex.AttributeDescriptions.len,

        .pVertexBindingDescriptions = &Vertex.BindingDescriptions,
        .pVertexAttributeDescriptions = &Vertex.AttributeDescriptions,
    };

    const inputAssembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .TRIANGLE_LIST,
        .primitiveRestartEnable = vk.FALSE,
    };

    const viewport = vk.Viewport{
        .x = 0.0,
        .y = 0.0,
        .width = @intToFloat(f32, swapChainExtent.width),
        .height = @intToFloat(f32, swapChainExtent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = swapChainExtent,
    };

    const viewportState = vk.PipelineViewportStateCreateInfo{
        .viewportCount = 1,
        .pViewports = arrayPtr(&viewport),
        .scissorCount = 1,
        .pScissors = arrayPtr(&scissor),
    };

    const rasterizer = vk.PipelineRasterizationStateCreateInfo{
        .depthClampEnable = vk.FALSE,
        .rasterizerDiscardEnable = vk.FALSE,
        .polygonMode = .FILL,
        .lineWidth = 1.0,
        .cullMode = .{ .back = true },
        .frontFace = .COUNTER_CLOCKWISE,
        .depthBiasEnable = vk.FALSE,

        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 0,
        .depthBiasSlopeFactor = 0,
    };

    const multisampling = vk.PipelineMultisampleStateCreateInfo{
        .sampleShadingEnable = vk.FALSE,
        .rasterizationSamples = .{ .t1 = true },
        .minSampleShading = 0,
        .pSampleMask = null,
        .alphaToCoverageEnable = 0,
        .alphaToOneEnable = 0,
    };

    const colorBlendAttachment = vk.PipelineColorBlendAttachmentState{
        .colorWriteMask = .{ .r = true, .g = true, .b = true, .a = true },
        .blendEnable = vk.FALSE,

        .srcColorBlendFactor = .ZERO,
        .dstColorBlendFactor = .ZERO,
        .colorBlendOp = .ADD,
        .srcAlphaBlendFactor = .ZERO,
        .dstAlphaBlendFactor = .ZERO,
        .alphaBlendOp = .ADD,
    };

    const colorBlending = vk.PipelineColorBlendStateCreateInfo{
        .logicOpEnable = vk.FALSE,
        .logicOp = .COPY,
        .attachmentCount = 1,
        .pAttachments = arrayPtr(&colorBlendAttachment),
        .blendConstants = .{ 0, 0, 0, 0 },
    };

    const pipelineLayoutInfo = vk.PipelineLayoutCreateInfo{
        .setLayoutCount = descriptorSetLayouts.len,
        .pSetLayouts = &descriptorSetLayouts,
    };

    pipelineLayout = try vk.CreatePipelineLayout(global_device, pipelineLayoutInfo, null);

    const pipelineInfo = vk.GraphicsPipelineCreateInfo{
        .stageCount = @intCast(u32, shaderStages.len),
        .pStages = &shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .layout = pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = null,

        .pTessellationState = null,
        .pDepthStencilState = null,
        .pDynamicState = null,
        .basePipelineIndex = 0,
    };

    try vk.CreateGraphicsPipelines(
        global_device,
        null,
        arrayPtr(&pipelineInfo),
        null,
        arrayPtr(&graphicsPipeline),
    );

    vk.DestroyShaderModule(global_device, fragShaderModule, null);
    vk.DestroyShaderModule(global_device, vertShaderModule, null);
}

fn createRenderPass() !void {
    const colorAttachment = vk.AttachmentDescription{
        .format = swapChainImageFormat,
        .samples = .{ .t1 = true },
        .loadOp = .CLEAR,
        .storeOp = .STORE,
        .stencilLoadOp = .DONT_CARE,
        .stencilStoreOp = .DONT_CARE,
        .initialLayout = .UNDEFINED,
        .finalLayout = .PRESENT_SRC_KHR,
    };

    const colorAttachmentRef = vk.AttachmentReference{
        .attachment = 0,
        .layout = .COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = vk.SubpassDescription{
        .pipelineBindPoint = .GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = arrayPtr(&colorAttachmentRef),
    };

    const dependency = vk.SubpassDependency{
        .srcSubpass = vk.SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = .{ .colorAttachmentOutput = true },
        .srcAccessMask = .{},
        .dstStageMask = .{ .colorAttachmentOutput = true },
        .dstAccessMask = .{ .colorAttachmentRead = true, .colorAttachmentWrite = true },
    };

    const renderPassInfo = vk.RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = arrayPtr(&colorAttachment),
        .subpassCount = 1,
        .pSubpasses = arrayPtr(&subpass),
        .dependencyCount = 1,
        .pDependencies = arrayPtr(&dependency),
    };

    renderPass = try vk.CreateRenderPass(global_device, renderPassInfo, null);
}

fn createDescriptorSetLayout() !void {
    const uboLayoutBindings = [_]vk.DescriptorSetLayoutBinding{.{
        .binding = 0,
        .descriptorType = .UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = .{ .vertex = true },
        .pImmutableSamplers = null,
    }};

    const layoutInfo = vk.DescriptorSetLayoutCreateInfo{
        .bindingCount = uboLayoutBindings.len,
        .pBindings = &uboLayoutBindings,
    };

    descriptorSetLayouts[0] = try vk.CreateDescriptorSetLayout(global_device, layoutInfo, null);
}

fn createImageViews(allocator: *Allocator) !void {
    swapChainImageViews = try allocator.alloc(vk.ImageView, swapChainImages.len);
    errdefer allocator.free(swapChainImageViews);

    for (swapChainImages) |swap_chain_image, i| {
        const createInfo = vk.ImageViewCreateInfo{
            .image = swap_chain_image,
            .viewType = .T_2D,
            .format = swapChainImageFormat,
            .components = .{
                .r = .IDENTITY,
                .g = .IDENTITY,
                .b = .IDENTITY,
                .a = .IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = .{ .color = true },
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        swapChainImageViews[i] = try vk.CreateImageView(global_device, createInfo, null);
    }
}

fn chooseSwapSurfaceFormat(availableFormats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    if (availableFormats.len == 1 and availableFormats[0].format == .UNDEFINED) {
        return .{
            .format = .B8G8R8A8_UNORM,
            .colorSpace = .SRGB_NONLINEAR,
        };
    }

    for (availableFormats) |availableFormat| {
        if (availableFormat.format == .B8G8R8A8_UNORM and
            availableFormat.colorSpace == .SRGB_NONLINEAR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

fn chooseSwapPresentMode(availablePresentModes: []vk.PresentModeKHR) vk.PresentModeKHR {
    var bestMode: vk.PresentModeKHR = .FIFO;

    for (availablePresentModes) |availablePresentMode| {
        if (availablePresentMode == .MAILBOX) {
            return availablePresentMode;
        } else if (availablePresentMode == .IMMEDIATE) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

fn chooseSwapExtent(capabilities: vk.SurfaceCapabilitiesKHR) vk.Extent2D {
    if (capabilities.currentExtent.width != maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        var actualExtent = vk.Extent2D{
            .width = WIDTH,
            .height = HEIGHT,
        };

        actualExtent.width = math.max(capabilities.minImageExtent.width, math.min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = math.max(capabilities.minImageExtent.height, math.min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

fn createSwapChain(allocator: *Allocator) !void {
    var swapChainSupport = try querySwapChainSupport(allocator, physicalDevice);
    defer swapChainSupport.deinit();

    const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats.items);
    const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes.items);
    const extent = chooseSwapExtent(swapChainSupport.capabilities);

    var imageCount: u32 = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 and
        imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const indices = try findQueueFamilies(allocator, physicalDevice);
    const queueFamilyIndices = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };

    const different_families = indices.graphicsFamily.? != indices.presentFamily.?;

    var createInfo = vk.SwapchainCreateInfoKHR{
        .surface = surface,

        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = .{ .colorAttachment = true },

        .imageSharingMode = if (different_families) .CONCURRENT else .EXCLUSIVE,
        .queueFamilyIndexCount = if (different_families) @as(u32, 2) else @as(u32, 0),
        .pQueueFamilyIndices = if (different_families) &queueFamilyIndices else &([_]u32{ 0, 0 }),

        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = .{ .opaque = true },
        .presentMode = presentMode,
        .clipped = vk.TRUE,

        .oldSwapchain = null,
    };

    swapChain = try vk.CreateSwapchainKHR(global_device, createInfo, null);

    imageCount = try vk.GetSwapchainImagesCountKHR(global_device, swapChain);
    swapChainImages = try allocator.alloc(vk.Image, imageCount);
    _ = try vk.GetSwapchainImagesKHR(global_device, swapChain, swapChainImages);

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

fn createLogicalDevice(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physicalDevice);

    var queueCreateInfos = std.ArrayList(vk.DeviceQueueCreateInfo).init(allocator);
    defer queueCreateInfos.deinit();
    const all_queue_families = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };
    const uniqueQueueFamilies = if (indices.graphicsFamily.? == indices.presentFamily.?)
        all_queue_families[0..1]
    else
        all_queue_families[0..2];

    var queuePriority: f32 = 1.0;
    for (uniqueQueueFamilies) |queueFamily| {
        const queueCreateInfo = vk.DeviceQueueCreateInfo{
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = arrayPtr(&queuePriority),
        };
        try queueCreateInfos.append(queueCreateInfo);
    }

    var deviceFeatures = mem.zeroes(vk.PhysicalDeviceFeatures);

    const createInfo = vk.DeviceCreateInfo{
        .queueCreateInfoCount = @intCast(u32, queueCreateInfos.items.len),
        .pQueueCreateInfos = queueCreateInfos.items.ptr,

        .pEnabledFeatures = &deviceFeatures,

        .enabledExtensionCount = @intCast(u32, deviceExtensions.len),
        .ppEnabledExtensionNames = &deviceExtensions,
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,
    };

    global_device = try vk.CreateDevice(physicalDevice, createInfo, null);

    graphicsQueue = vk.GetDeviceQueue(global_device, indices.graphicsFamily.?, 0);
    presentQueue = vk.GetDeviceQueue(global_device, indices.presentFamily.?, 0);
}

fn pickPhysicalDevice(allocator: *Allocator) !void {
    var deviceCount = try vk.EnumeratePhysicalDevicesCount(instance);

    if (deviceCount == 0) {
        return error.FailedToFindGPUsWithVulkanSupport;
    }

    const devicesBuf = try allocator.alloc(vk.PhysicalDevice, deviceCount);
    defer allocator.free(devicesBuf);

    var devices = (try vk.EnumeratePhysicalDevices(instance, devicesBuf)).physicalDevices;

    physicalDevice = for (devices) |device| {
        if (try isDeviceSuitable(allocator, device)) {
            break device;
        }
    } else return error.FailedToFindSuitableGPU;
}

fn findQueueFamilies(allocator: *Allocator, device: vk.PhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices.init();

    var queueFamilyCount = vk.GetPhysicalDeviceQueueFamilyPropertiesCount(device);

    const queueFamiliesBuf = try allocator.alloc(vk.QueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamiliesBuf);

    var queueFamilies = vk.GetPhysicalDeviceQueueFamilyProperties(device, queueFamiliesBuf);

    var i: u32 = 0;
    for (queueFamilies) |queueFamily| {
        if (queueFamily.queueCount > 0 and
            queueFamily.queueFlags.graphics)
        {
            indices.graphicsFamily = i;
        }

        var presentSupport = try vk.GetPhysicalDeviceSurfaceSupportKHR(device, i, surface);

        if (queueFamily.queueCount > 0 and presentSupport != 0) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i += 1;
    }

    return indices;
}

fn isDeviceSuitable(allocator: *Allocator, device: vk.PhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, device);

    const extensionsSupported = try checkDeviceExtensionSupport(allocator, device);

    var swapChainAdequate = false;
    if (extensionsSupported) {
        var swapChainSupport = try querySwapChainSupport(allocator, device);
        defer swapChainSupport.deinit();
        swapChainAdequate = swapChainSupport.formats.items.len != 0 and swapChainSupport.presentModes.items.len != 0;
    }

    return indices.isComplete() and extensionsSupported and swapChainAdequate;
}

fn querySwapChainSupport(allocator: *Allocator, device: vk.PhysicalDevice) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails.init(allocator);

    details.capabilities = try vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface);

    var formatCount = try vk.GetPhysicalDeviceSurfaceFormatsCountKHR(device, surface);
    if (formatCount != 0) {
        try details.formats.resize(formatCount);
        _ = try vk.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, details.formats.items);
    }

    var presentModeCount = try vk.GetPhysicalDeviceSurfacePresentModesCountKHR(device, surface);
    if (presentModeCount != 0) {
        try details.presentModes.resize(presentModeCount);
        _ = try vk.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, details.presentModes.items);
    }

    return details;
}

fn checkDeviceExtensionSupport(allocator: *Allocator, device: vk.PhysicalDevice) !bool {
    var extensionCount = try vk.EnumerateDeviceExtensionPropertiesCount(device, null);

    const availableExtensionsBuf = try allocator.alloc(vk.ExtensionProperties, extensionCount);
    defer allocator.free(availableExtensionsBuf);

    var availableExtensions = (try vk.EnumerateDeviceExtensionProperties(device, null, availableExtensionsBuf)).properties;

    var requiredExtensions = std.HashMap([*:0]const u8, void, hash_cstr, eql_cstr).init(allocator);
    defer requiredExtensions.deinit();
    for (deviceExtensions) |device_ext| {
        _ = try requiredExtensions.put(device_ext, {});
    }

    for (availableExtensions) |extension| {
        _ = requiredExtensions.remove(&extension.extensionName);
    }

    return requiredExtensions.count() == 0;
}

fn createSurface(window: *glfw.GLFWwindow) !void {
    if (glfw.glfwCreateWindowSurface(instance, window, null, &surface) != vk.Result.SUCCESS) {
        return error.FailedToCreateWindowSurface;
    }
}

fn debugCallback(
    flags: vk.DebugReportFlagsEXT.IntType,
    objType: vk.DebugReportObjectTypeEXT,
    obj: u64,
    location: usize,
    code: i32,
    layerPrefix: ?vk.CString,
    msg: ?vk.CString,
    userData: ?*c_void,
) callconv(vk.CallConv) vk.Bool32 {
    std.debug.warn("validation layer: {s}\n", .{msg});
    return vk.FALSE;
}

fn setupDebugCallback() error{FailedToSetUpDebugCallback}!void {
    if (!enableValidationLayers) return;

    var createInfo = vk.DebugReportCallbackCreateInfoEXT{
        .flags = .{ .errorBit = true, .warning = true },
        .pfnCallback = debugCallback,
        .pUserData = null,
    };

    if (CreateDebugReportCallbackEXT(&createInfo, null, &callback) != .SUCCESS) {
        return error.FailedToSetUpDebugCallback;
    }
}

fn DestroyDebugReportCallbackEXT(
    pAllocator: ?*const vk.AllocationCallbacks,
) void {
    const func = @ptrCast(?@TypeOf(vk.vkDestroyDebugReportCallbackEXT), vk.GetInstanceProcAddr(
        instance,
        "vkDestroyDebugReportCallbackEXT",
    )) orelse unreachable;
    func(instance, callback, pAllocator);
}

fn CreateDebugReportCallbackEXT(
    pCreateInfo: *const vk.DebugReportCallbackCreateInfoEXT,
    pAllocator: ?*const vk.AllocationCallbacks,
    pCallback: *vk.DebugReportCallbackEXT,
) vk.Result {
    const func = @ptrCast(?@TypeOf(vk.vkCreateDebugReportCallbackEXT), vk.GetInstanceProcAddr(
        instance,
        "vkCreateDebugReportCallbackEXT",
    )) orelse return .ERROR_EXTENSION_NOT_PRESENT;
    return func(instance, pCreateInfo, pAllocator, pCallback);
}

fn createInstance(allocator: *Allocator) !void {
    if (enableValidationLayers) {
        if (!(try checkValidationLayerSupport(allocator))) {
            return error.ValidationLayerRequestedButNotAvailable;
        }
    }

    const appInfo = vk.ApplicationInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = vk.MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = vk.MAKE_VERSION(1, 0, 0),
        .apiVersion = vk.API_VERSION_1_0,
    };

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    const createInfo = vk.InstanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = @intCast(u32, extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,
    };

    instance = try vk.CreateInstance(createInfo, null);
}

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![]vk.CString {
    var glfwExtensionCount: u32 = 0;
    var glfwExtensions: [*]const vk.CString = glfw.glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    var extensions = std.ArrayList(vk.CString).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfwExtensions[0..glfwExtensionCount]);

    if (enableValidationLayers) {
        try extensions.append(vk.EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount = try vk.EnumerateInstanceLayerPropertiesCount();

    const layersBuffer = try allocator.alloc(vk.LayerProperties, layerCount);
    defer allocator.free(layersBuffer);

    var availableLayers = (try vk.EnumerateInstanceLayerProperties(layersBuffer)).properties;

    for (validationLayers) |layerName| {
        var layerFound = false;

        for (availableLayers) |layerProperties| {
            if (std.cstr.cmp(layerName, &layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

fn drawFrame() !void {
    _ = try vk.WaitForFences(global_device, inFlightFences[currentFrame .. currentFrame + 1], vk.TRUE, maxInt(u64));
    try vk.ResetFences(global_device, inFlightFences[currentFrame .. currentFrame + 1]);

    var imageIndex = (try vk.AcquireNextImageKHR(global_device, swapChain, maxInt(u64), imageAvailableSemaphores[currentFrame], null)).imageIndex;

    try updateUniformBuffer(imageIndex);

    var waitSemaphores = [_]vk.Semaphore{imageAvailableSemaphores[currentFrame]};
    var waitStages: [1]vk.PipelineStageFlags align(4) = .{.{ .colorAttachmentOutput = true }};

    const signalSemaphores = [_]vk.Semaphore{renderFinishedSemaphores[currentFrame]};

    var submitInfo = [_]vk.SubmitInfo{vk.SubmitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSemaphores,
        .pWaitDstStageMask = &waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = commandBuffers.ptr + imageIndex,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphores,
    }};

    try vk.QueueSubmit(graphicsQueue, &submitInfo, inFlightFences[currentFrame]);

    const swapChains = [_]vk.SwapchainKHR{swapChain};
    const presentInfo = vk.PresentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signalSemaphores,

        .swapchainCount = 1,
        .pSwapchains = &swapChains,

        .pImageIndices = arrayPtr(&imageIndex),

        .pResults = null,
    };

    _ = try vk.QueuePresentKHR(presentQueue, presentInfo);

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

var hasDumped = false;

fn updateUniformBuffer(index: u32) !void {
    const time = currentTime();

    var ubo = UniformBufferObject{
        .model = Mat4.Identity,
        .view = Mat4.Identity,
        .proj = Mat4.Identity,
    };

    geo.Generic.setMat3(&ubo.model, Rotor3.aroundZ(time * math.pi * 0.5).toMat3());
    ubo.proj = geo.symmetricOrtho(1, -@as(f32, HEIGHT) / @as(f32, WIDTH), -1, 1);

    if (!hasDumped) {
        std.debug.warn("proj = {}\n", .{ubo.proj});
        hasDumped = true;
    }

    var data: *c_void = undefined;
    try vk.MapMemory(global_device, uniformBuffersMemory[index], 0, @sizeOf(UniformBufferObject), .{}, &data);
    @memcpy(@ptrCast([*]u8, data), @ptrCast([*]const u8, &ubo), @sizeOf(UniformBufferObject));
    vk.UnmapMemory(global_device, uniformBuffersMemory[index]);
}

fn hash_cstr(a: [*:0]const u8) u32 {
    // FNV 32-bit hash
    var h: u32 = 2166136261;
    var i: usize = 0;
    while (a[i] != 0) : (i += 1) {
        h ^= a[i];
        h *%= 16777619;
    }
    return h;
}

fn eql_cstr(a: [*:0]const u8, b: [*:0]const u8) bool {
    return std.cstr.cmp(a, b) == 0;
}
