#include "application.hpp"
#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

namespace {

struct Vec3 { float x, y, z; };

struct Mat4 {
    float m[16];

    static Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 translate(Vec3 v) {
        Mat4 r = identity();
        r.m[12] = v.x;
        r.m[13] = v.y;
        r.m[14] = v.z;
        return r;
    }

    static Mat4 scale(Vec3 v) {
        Mat4 r = identity();
        r.m[0] = v.x;
        r.m[5] = v.y;
        r.m[10] = v.z;
        return r;
    }

    static Mat4 rotateX(float a) {
        Mat4 r = identity();
        r.m[5] = cosf(a);
        r.m[6] = sinf(a);
        r.m[9] = -sinf(a);
        r.m[10] = cosf(a);
        return r;
    }

    static Mat4 rotateZ(float a) {
        Mat4 r = identity();
        r.m[0] = cosf(a);
        r.m[1] = sinf(a);
        r.m[4] = -sinf(a);
        r.m[5] = cosf(a);
        return r;
    }

    static Mat4 rotateY(float a) {
        Mat4 r = identity();
        r.m[0] = cosf(a);
        r.m[2] = -sinf(a);
        r.m[8] = sinf(a);
        r.m[10] = cosf(a);
        return r;
    }

    static Mat4 perspective(float fov, float aspect, float near, float far) {
        float tanHalf = tanf(fov * 0.5f);
        Mat4 r{};
        r.m[0] = 1.0f / (aspect * tanHalf);
        r.m[5] = 1.0f / tanHalf;
        r.m[10] = far / (far - near);
        r.m[11] = 1.0f;
        r.m[14] = -far * near / (far - near);
        return r;
    }

    static Mat4 orthographic(float left, float right, float bottom, float top, float near, float far) {
        Mat4 r{};
        r.m[0]  = 2.0f / (right - left);
        r.m[5]  = 2.0f / (top - bottom);
        r.m[10] = 1.0f / (far - near);
        r.m[12] = -(right + left) / (right - left);
        r.m[13] = -(top + bottom) / (top - bottom);
        r.m[14] = -near / (far - near);
        r.m[15] = 1.0f;
        return r;
    }

    static Mat4 inverseAffine(const Mat4& m) {
        Mat4 r{};
        r.m[0] = m.m[0];  r.m[4] = m.m[1];  r.m[8]  = m.m[2];
        r.m[1] = m.m[4];  r.m[5] = m.m[5];  r.m[9]  = m.m[6];
        r.m[2] = m.m[8];  r.m[6] = m.m[9];  r.m[10] = m.m[10];

        float tx = m.m[12], ty = m.m[13], tz = m.m[14];
        r.m[12] = -(r.m[0] * tx + r.m[4] * ty + r.m[8]  * tz);
        r.m[13] = -(r.m[1] * tx + r.m[5] * ty + r.m[9]  * tz);
        r.m[14] = -(r.m[2] * tx + r.m[6] * ty + r.m[10] * tz);

        r.m[3] = 0.0f; r.m[7] = 0.0f; r.m[11] = 0.0f;
        r.m[15] = 1.0f;
        return r;
    }
};

struct Vertex {
    float position[3];
    float normal[3];
    float color[3];
};

struct SceneUniforms {
    float view_projection[4][4];
    float camera_position[4];
    float light_direction[4];
    float light_color[4];
    float ambient[4];
    int point_light_count;
    int spot_light_count;
    int _pad5;
    int _pad6;
};

struct ModelUniforms {
    float model[4][4];
    float albedo[3];
    float _pad1;
    float specular[3];
    float shininess;
};

constexpr int MAX_LIGHTS = 16;

struct PointLight {
    float position[4];
    float color[4];
};

struct SpotLight {
    float position[4];
    float color[4];
    float direction[4];
    float inner_angle;
    float outer_angle;
    float range;
    float _pad;
};

VkPipeline vk_pipeline = VK_NULL_HANDLE;
VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;

VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;
Vertex* vk_vertex_buffer_memory = nullptr;
uint32_t* vk_index_buffer_memory = nullptr;

VkBuffer vk_index_buffer = VK_NULL_HANDLE;
VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

VkBuffer vk_uniform_buffer_scene = VK_NULL_HANDLE;
VmaAllocation vk_uniform_buffer_scene_allocation = VK_NULL_HANDLE;
SceneUniforms* vk_uniform_buffer_scene_memory = nullptr;

VkBuffer vk_uniform_buffer_model = VK_NULL_HANDLE;
VmaAllocation vk_uniform_buffer_model_allocation = VK_NULL_HANDLE;
ModelUniforms* vk_uniform_buffer_model_memory = nullptr;

VkBuffer vk_buffer_point_lights = VK_NULL_HANDLE;
VmaAllocation vk_buffer_point_lights_allocation = VK_NULL_HANDLE;
PointLight* vk_buffer_point_lights_memory = nullptr;

VkBuffer vk_buffer_spot_lights = VK_NULL_HANDLE;
VmaAllocation vk_buffer_spot_lights_allocation = VK_NULL_HANDLE;
SpotLight* vk_buffer_spot_lights_memory = nullptr;

VkDescriptorSetLayout vk_descriptor_set_layout_scene = VK_NULL_HANDLE;
VkDescriptorSetLayout vk_descriptor_set_layout_model = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_descriptor_set_scene = VK_NULL_HANDLE;
VkDescriptorSet vk_descriptor_set_model = VK_NULL_HANDLE;

VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

uint32_t index_count = 0;

float pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;
float rot_x = 0.0f, rot_y = 0.0f, rot_z = 0.0f;
float scale_x = 1.0f, scale_y = 1.0f, scale_z = 1.0f;

bool use_perspective = true;
float fov_degrees = 45.0f;

float light_dir_x = -0.5f, light_dir_y = -1.0f, light_dir_z = -0.5f;
float light_color[3] = { 1.0f, 1.0f, 1.0f };
float light_intensity = 1.0f;
float ambient[3] = { 0.15f, 0.15f, 0.2f };

float albedo[3] = { 0.8f, 0.6f, 0.4f };
float specular[3] = { 1.0f, 1.0f, 1.0f };
float shininess = 32.0f;

PointLight point_lights[MAX_LIGHTS] = {};
int point_light_count = 0;

SpotLight spot_lights[MAX_LIGHTS] = {};
int spot_light_count = 0;

bool animating = false;
float anim_speed = 1.0f;
float anim_radius = 2.0f;
float anim_time = 0.0f;
double last_time = 0.0;
int anim_mode = 0;

float cam_x = 0.0f, cam_y = 0.0f, cam_z = -5.0f;
float cam_yaw = 0.0f;
float cam_pitch = 0.0f;

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

void generateOctahedron() {
    Vec3 positions[6] = {
        { +1.0f,  0.0f,  0.0f },
        { -1.0f,  0.0f,  0.0f },
        {  0.0f, +1.0f,  0.0f },
        {  0.0f, -1.0f,  0.0f },
        {  0.0f,  0.0f, +1.0f },
        {  0.0f,  0.0f, -1.0f },
    };

    const int faces[8][3] = {
        {2, 0, 4}, {2, 4, 1}, {2, 1, 5}, {2, 5, 0},
        {3, 0, 5}, {3, 5, 1}, {3, 1, 4}, {3, 4, 0},
    };

    Vertex verts[24]{};
    uint32_t indices[24];
    index_count = 24;

    for (int f = 0; f < 8; ++f) {
        Vec3 a = positions[faces[f][0]];
        Vec3 b = positions[faces[f][1]];
        Vec3 c = positions[faces[f][2]];

        float nx = a.x + b.x + c.x;
        float ny = a.y + b.y + c.y;
        float nz = a.z + b.z + c.z;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        nx /= len; ny /= len; nz /= len;

        for (int k = 0; k < 3; ++k) {
            int idx = f * 3 + k;
            Vec3 p = positions[faces[f][k]];
            verts[idx].position[0] = p.x;
            verts[idx].position[1] = p.y;
            verts[idx].position[2] = p.z;
            verts[idx].normal[0] = nx;
            verts[idx].normal[1] = ny;
            verts[idx].normal[2] = nz;
            verts[idx].color[0] = p.x * 0.5f + 0.5f;
            verts[idx].color[1] = p.y * 0.5f + 0.5f;
            verts[idx].color[2] = p.z * 0.5f + 0.5f;
            indices[idx] = idx;
        }
    }

    const VkBufferCreateInfo vertex_buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(verts),
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo vertex_buffer_allocation = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (vmaCreateBuffer(graphics::internal::context.allocator,
                        &vertex_buffer_info, &vertex_buffer_allocation,
                        &vk_vertex_buffer, &vk_vertex_buffer_allocation,
                        nullptr) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create vertex buffer\n");
        return;
    }
    if (vmaMapMemory(graphics::internal::context.allocator,
                     vk_vertex_buffer_allocation,
                     (void**)&vk_vertex_buffer_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map vertex buffer memory\n");
        return;
    }
    memcpy(vk_vertex_buffer_memory, verts, sizeof(verts));

    const VkBufferCreateInfo index_buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(indices),
        .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo index_buffer_allocation = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (vmaCreateBuffer(graphics::internal::context.allocator,
                        &index_buffer_info, &index_buffer_allocation,
                        &vk_index_buffer, &vk_index_buffer_allocation,
                        nullptr) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create index buffer\n");
        return;
    }
    if (vmaMapMemory(graphics::internal::context.allocator,
                     vk_index_buffer_allocation,
                     (void**)&vk_index_buffer_memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map index buffer memory\n");
        return;
    }
    memcpy(vk_index_buffer_memory, indices, sizeof(indices));
}

VkShaderModule loadShaderModule(const char path[]) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device,
                             &info, nullptr, &result) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return result;
}

bool createPipelineLayout() {
    const VkDescriptorSetLayout layouts[] = {
        vk_descriptor_set_layout_scene,
        vk_descriptor_set_layout_model,
    };
    const VkPipelineLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = layouts,
    };
    if (vkCreatePipelineLayout(graphics::internal::context.device,
                               &info, nullptr,
                               &vk_pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create pipeline layout\n");
        return false;
    }
    return true;
}

bool createDescriptorSetLayouts() {
    {
        const VkDescriptorSetLayoutBinding bindings[] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        };
        const VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };
        if (vkCreateDescriptorSetLayout(graphics::internal::context.device,
                                        &info, nullptr,
                                        &vk_descriptor_set_layout_scene) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create scene descriptor set layout\n");
            return false;
        }
    }

    {
        const VkDescriptorSetLayoutBinding binding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        if (vkCreateDescriptorSetLayout(graphics::internal::context.device,
                                        &info, nullptr,
                                        &vk_descriptor_set_layout_model) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create model descriptor set layout\n");
            return false;
        }
    }

    return true;
}

bool createPipeline() {
    vk_vertex_shader = loadShaderModule("shaders/octahedron.vert.spv");
    vk_fragment_shader = loadShaderModule("shaders/octahedron.frag.spv");
    if (vk_vertex_shader == VK_NULL_HANDLE || vk_fragment_shader == VK_NULL_HANDLE) {
        fprintf(stderr, "Failed to load shaders\n");
        return false;
    }

    const VkPipelineShaderStageCreateInfo shader_stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vk_vertex_shader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = vk_fragment_shader,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription vertex_bindings[] = {
        {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        },
    };

    const VkVertexInputAttributeDescription vertex_attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
        {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = vertex_bindings,
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = vertex_attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = &depth_stencil_state,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = &dynamic_state,
        .layout = vk_pipeline_layout,
        .renderPass = graphics::internal::context.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(graphics::internal::context.device,
                                  VK_NULL_HANDLE,
                                  1, &pipeline_info,
                                  nullptr, &vk_pipeline) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create graphics pipeline\n");
        return false;
    }
    return true;
}

bool createUniformBuffersAndSets() {
    {
        const VkBufferCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(SceneUniforms),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(graphics::internal::context.allocator,
                            &info, &alloc,
                            &vk_uniform_buffer_scene,
                            &vk_uniform_buffer_scene_allocation,
                            nullptr) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create scene buffer\n");
            return false;
        }
        if (vmaMapMemory(graphics::internal::context.allocator,
                         vk_uniform_buffer_scene_allocation,
                         (void**)&vk_uniform_buffer_scene_memory) != VK_SUCCESS) {
            fprintf(stderr, "Failed to map scene buffer\n");
            return false;
        }
    }

    {
        const VkBufferCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(ModelUniforms),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(graphics::internal::context.allocator,
                            &info, &alloc,
                            &vk_uniform_buffer_model,
                            &vk_uniform_buffer_model_allocation,
                            nullptr) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create model buffer\n");
            return false;
        }
        if (vmaMapMemory(graphics::internal::context.allocator,
                         vk_uniform_buffer_model_allocation,
                         (void**)&vk_uniform_buffer_model_memory) != VK_SUCCESS) {
            fprintf(stderr, "Failed to map model buffer\n");
            return false;
        }
    }

    {
        const VkBufferCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(PointLight) * MAX_LIGHTS,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(graphics::internal::context.allocator,
                            &info, &alloc,
                            &vk_buffer_point_lights,
                            &vk_buffer_point_lights_allocation,
                            nullptr) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create point lights buffer\n");
            return false;
        }
        if (vmaMapMemory(graphics::internal::context.allocator,
                         vk_buffer_point_lights_allocation,
                         (void**)&vk_buffer_point_lights_memory) != VK_SUCCESS) {
            fprintf(stderr, "Failed to map point lights buffer\n");
            return false;
        }
    }

    {
        const VkBufferCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(SpotLight) * MAX_LIGHTS,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(graphics::internal::context.allocator,
                            &info, &alloc,
                            &vk_buffer_spot_lights,
                            &vk_buffer_spot_lights_allocation,
                            nullptr) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create spot lights buffer\n");
            return false;
        }
        if (vmaMapMemory(graphics::internal::context.allocator,
                         vk_buffer_spot_lights_allocation,
                         (void**)&vk_buffer_spot_lights_memory) != VK_SUCCESS) {
            fprintf(stderr, "Failed to map spot lights buffer\n");
            return false;
        }
    }

    const VkDescriptorPoolSize sizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = sizes,
    };
    if (vkCreateDescriptorPool(graphics::internal::context.device,
                               &pool_info, nullptr,
                               &vk_descriptor_pool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
        return false;
    }

    {
        const VkDescriptorSetAllocateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout_scene,
        };
        if (vkAllocateDescriptorSets(graphics::internal::context.device,
                                     &info, &vk_descriptor_set_scene) != VK_SUCCESS) {
            fprintf(stderr, "Failed to allocate scene set\n");
            return false;
        }
        const VkDescriptorBufferInfo scene_info = {
            .buffer = vk_uniform_buffer_scene,
            .offset = 0,
            .range = sizeof(SceneUniforms),
        };
        const VkDescriptorBufferInfo point_info = {
            .buffer = vk_buffer_point_lights,
            .offset = 0,
            .range = sizeof(PointLight) * MAX_LIGHTS,
        };
        const VkDescriptorBufferInfo spot_info = {
            .buffer = vk_buffer_spot_lights,
            .offset = 0,
            .range = sizeof(SpotLight) * MAX_LIGHTS,
        };
        const VkWriteDescriptorSet writes[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_descriptor_set_scene,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &scene_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_descriptor_set_scene,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &point_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_descriptor_set_scene,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &spot_info,
            },
        };
        vkUpdateDescriptorSets(graphics::internal::context.device, 3, writes, 0, nullptr);
    }

    {
        const VkDescriptorSetAllocateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout_model,
        };
        if (vkAllocateDescriptorSets(graphics::internal::context.device,
                                     &info, &vk_descriptor_set_model) != VK_SUCCESS) {
            fprintf(stderr, "Failed to allocate model set\n");
            return false;
        }
        const VkDescriptorBufferInfo buffer_info = {
            .buffer = vk_uniform_buffer_model,
            .offset = 0,
            .range = sizeof(ModelUniforms),
        };
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_descriptor_set_model,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
    }

    return true;
}

} // namespace

namespace application {

bool initialize() {
    generateOctahedron();
    if (!createDescriptorSetLayouts()) return false;
    if (!createPipelineLayout()) return false;
    if (!createUniformBuffersAndSets()) return false;
    if (!createPipeline()) return false;
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    if (vk_pipeline) vkDestroyPipeline(context.device, vk_pipeline, nullptr);
    if (vk_pipeline_layout) vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
    if (vk_descriptor_pool) vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
    if (vk_descriptor_set_layout_scene) vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout_scene, nullptr);
    if (vk_descriptor_set_layout_model) vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout_model, nullptr);
    if (vk_vertex_shader) vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);
    if (vk_fragment_shader) vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);

    if (vk_vertex_buffer) vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
    if (vk_index_buffer) vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
    if (vk_uniform_buffer_scene) vmaDestroyBuffer(context.allocator, vk_uniform_buffer_scene, vk_uniform_buffer_scene_allocation);
    if (vk_uniform_buffer_model) vmaDestroyBuffer(context.allocator, vk_uniform_buffer_model, vk_uniform_buffer_model_allocation);
    if (vk_buffer_point_lights) vmaDestroyBuffer(context.allocator, vk_buffer_point_lights, vk_buffer_point_lights_allocation);
    if (vk_buffer_spot_lights) vmaDestroyBuffer(context.allocator, vk_buffer_spot_lights, vk_buffer_spot_lights_allocation);
}

void update(double time) {
    const float dt = (float)(time - last_time);
    last_time = time;

    ImGui::SetNextWindowSize(ImVec2(380, 600), ImGuiCond_Once);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar);

    ImGui::Checkbox("Perspective", &use_perspective);
    if (use_perspective) ImGui::SliderFloat("FOV", &fov_degrees, 20.0f, 120.0f);
    ImGui::Separator();

    ImGui::SliderFloat("Pos X", &pos_x, -3.0f, 3.0f);
    ImGui::SliderFloat("Pos Y", &pos_y, -3.0f, 3.0f);
    ImGui::SliderFloat("Pos Z", &pos_z, -3.0f, 3.0f);
    ImGui::SliderFloat("Rot X", &rot_x, -3.14f, 3.14f);
    ImGui::SliderFloat("Rot Y", &rot_y, -3.14f, 3.14f);
    ImGui::SliderFloat("Rot Z", &rot_z, -3.14f, 3.14f);
    ImGui::SliderFloat("Scale X", &scale_x, 0.1f, 3.0f);
    ImGui::SliderFloat("Scale Y", &scale_y, 0.1f, 3.0f);
    ImGui::SliderFloat("Scale Z", &scale_z, 0.1f, 3.0f);
    ImGui::Separator();

    ImGui::Text("Light");
    ImGui::SliderFloat("Light Dir X", &light_dir_x, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Y", &light_dir_y, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Z", &light_dir_z, -1.0f, 1.0f);
    ImGui::SliderFloat("Intensity", &light_intensity, 0.0f, 5.0f);
    ImGui::ColorEdit3("Ambient", ambient);
    ImGui::Separator();

    ImGui::Text("Point Lights (%d)", point_light_count);
    if (ImGui::Button("+ Add Point Light") && point_light_count < MAX_LIGHTS) {
        PointLight& pl = point_lights[point_light_count];
        pl.position[0] = 0.8f;
        pl.position[1] = 0.8f;
        pl.position[2] = 0.8f;
        pl.position[3] = 0.0f;
        pl.color[0] = 15.0f;
        pl.color[1] = 8.0f;
        pl.color[2] = 8.0f;
        pl.color[3] = 0.0f;
        point_light_count++;
    }
    ImGui::SameLine();
    if (ImGui::Button("- Remove Point Light") && point_light_count > 0) {
        point_light_count--;
    }
    for (int i = 0; i < point_light_count; ++i) {
        ImGui::PushID(i);
        ImGui::Text("Point %d", i);
        ImGui::SliderFloat("Pos X", &point_lights[i].position[0], -10.0f, 10.0f);
        ImGui::SliderFloat("Pos Y", &point_lights[i].position[1], -10.0f, 10.0f);
        ImGui::SliderFloat("Pos Z", &point_lights[i].position[2], -10.0f, 10.0f);
        ImGui::ColorEdit3("Color", point_lights[i].color);
        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::Text("Spot Lights (%d)", spot_light_count);
    if (ImGui::Button("+ Add Spot Light") && spot_light_count < MAX_LIGHTS) {
        SpotLight& sl = spot_lights[spot_light_count];
        sl.position[0] = 0.0f;
        sl.position[1] = 1.5f;
        sl.position[2] = 0.0f;
        sl.position[3] = 0.0f;
        sl.color[0] = 10.0f;
        sl.color[1] = 10.0f;
        sl.color[2] = 20.0f;
        sl.color[3] = 0.0f;
        sl.direction[0] = 0.0f;
        sl.direction[1] = -1.0f;
        sl.direction[2] = 0.0f;
        sl.direction[3] = 0.0f;
        sl.inner_angle = 0.85f;
        sl.outer_angle = 0.3f;
        sl.range = 20.0f;
        sl._pad = 0.0f;
        spot_light_count++;
    }
    ImGui::SameLine();
    if (ImGui::Button("- Remove Spot Light") && spot_light_count > 0) {
        spot_light_count--;
    }
    for (int i = 0; i < spot_light_count; ++i) {
        ImGui::PushID(1000 + i);
        ImGui::Text("Spot %d", i);
        ImGui::SliderFloat("Pos X", &spot_lights[i].position[0], -10.0f, 10.0f);
        ImGui::SliderFloat("Pos Y", &spot_lights[i].position[1], -10.0f, 10.0f);
        ImGui::SliderFloat("Pos Z", &spot_lights[i].position[2], -10.0f, 10.0f);
        ImGui::SliderFloat("Dir X", &spot_lights[i].direction[0], -1.0f, 1.0f);
        ImGui::SliderFloat("Dir Y", &spot_lights[i].direction[1], -1.0f, 1.0f);
        ImGui::SliderFloat("Dir Z", &spot_lights[i].direction[2], -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", spot_lights[i].color);
        ImGui::SliderFloat("Inner (cos)", &spot_lights[i].inner_angle, 0.0f, 1.0f);
        ImGui::SliderFloat("Outer (cos)", &spot_lights[i].outer_angle, 0.0f, 1.0f);
        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::Text("Material");
    ImGui::ColorEdit3("Albedo", albedo);
    ImGui::ColorEdit3("Specular", specular);
    ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
    ImGui::Separator();

    ImGui::Text("Camera");
    ImGui::SliderFloat("Cam X", &cam_x, -10.0f, 10.0f);
    ImGui::SliderFloat("Cam Y", &cam_y, -10.0f, 10.0f);
    ImGui::SliderFloat("Cam Z", &cam_z, -10.0f, 10.0f);
    ImGui::SliderFloat("Yaw", &cam_yaw, -3.14f, 3.14f);
    ImGui::SliderFloat("Pitch", &cam_pitch, -1.4f, 1.4f);
    ImGui::Separator();

    ImGui::Checkbox("Animate", &animating);
    ImGui::Text("Trajectory:");
    ImGui::RadioButton("Circle", &anim_mode, 0); ImGui::SameLine();
    ImGui::RadioButton("Lissajous", &anim_mode, 1);
    ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 5.0f);
    ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 5.0f);
    ImGui::End();

    if (animating) {
        anim_time += dt * anim_speed;

        if (anim_mode == 0) {
            pos_x = cosf(anim_time) * anim_radius;
            pos_y = 0.0f;
            pos_z = sinf(anim_time) * anim_radius;
        } else {
            pos_x = sinf(2.0f * anim_time) * anim_radius;
            pos_y = sinf(3.0f * anim_time) * anim_radius * 0.3f;
            pos_z = sinf(1.0f * anim_time) * anim_radius;
        }

        rot_x += dt * anim_speed * 0.7f;
        rot_y += dt * anim_speed * 1.3f;
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;
    VkCommandBuffer cb = fd.command_buffer;

    vkResetCommandBuffer(cb, 0);

    const VkCommandBufferBeginInfo command_buffer_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &command_buffer_begin);

    Mat4 model = Mat4::identity();
    model = multiply(model, Mat4::translate({ pos_x, pos_y, pos_z }));
    model = multiply(model, Mat4::rotateX(rot_x));
    model = multiply(model, Mat4::rotateY(rot_y));
    model = multiply(model, Mat4::rotateZ(rot_z));
    model = multiply(model, Mat4::scale({ scale_x, scale_y, scale_z }));

    Mat4 cam_world = multiply(
        Mat4::translate({ cam_x, cam_y, cam_z }),
        multiply(Mat4::rotateY(cam_yaw), Mat4::rotateX(cam_pitch))
    );
    Mat4 view = Mat4::inverseAffine(cam_world);

    const float aspect = (float)context.swapchain_extent.width /
                         (float)context.swapchain_extent.height;
    Mat4 proj;
    if (use_perspective) {
        proj = Mat4::perspective(fov_degrees * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);
    } else {
        float s = 2.0f;
        proj = Mat4::orthographic(-s * aspect, s * aspect, -s, s, 0.1f, 100.0f);
    }

    const Mat4 mvp = multiply(proj, multiply(view, model));

    memcpy(vk_uniform_buffer_scene_memory->view_projection, mvp.m, sizeof(mvp.m));

    vk_uniform_buffer_scene_memory->camera_position[0] = cam_x;
    vk_uniform_buffer_scene_memory->camera_position[1] = cam_y;
    vk_uniform_buffer_scene_memory->camera_position[2] = cam_z;
    vk_uniform_buffer_scene_memory->camera_position[3] = 0.0f;

    vk_uniform_buffer_scene_memory->light_direction[0] = light_dir_x;
    vk_uniform_buffer_scene_memory->light_direction[1] = light_dir_y;
    vk_uniform_buffer_scene_memory->light_direction[2] = light_dir_z;
    vk_uniform_buffer_scene_memory->light_direction[3] = 0.0f;

    for (int i = 0; i < 3; ++i) {
        vk_uniform_buffer_scene_memory->light_color[i] = light_color[i] * light_intensity;
        vk_uniform_buffer_scene_memory->ambient[i] = ambient[i];
    }
    vk_uniform_buffer_scene_memory->light_color[3] = 0.0f;
    vk_uniform_buffer_scene_memory->ambient[3] = 0.0f;

    vk_uniform_buffer_scene_memory->point_light_count = point_light_count;
    vk_uniform_buffer_scene_memory->spot_light_count = spot_light_count;

    if (point_light_count > 0) {
        memcpy(vk_buffer_point_lights_memory, point_lights,
               sizeof(PointLight) * point_light_count);
    }

    if (spot_light_count > 0) {
        memcpy(vk_buffer_spot_lights_memory, spot_lights,
               sizeof(SpotLight) * spot_light_count);
    }

    memcpy(vk_uniform_buffer_model_memory->model, model.m, sizeof(model.m));
    for (int i = 0; i < 3; ++i) {
        vk_uniform_buffer_model_memory->albedo[i] = albedo[i];
        vk_uniform_buffer_model_memory->specular[i] = specular[i];
    }
    vk_uniform_buffer_model_memory->shininess = shininess;

    const VkClearValue clear_values[] = {
        { .color = { .float32 = { 0.05f, 0.05f, 0.1f, 1.0f } } },
        { .depthStencil = { 1.0f, 0 } },
    };
    const VkRenderPassBeginInfo render_pass_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(cb, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)context.swapchain_extent.width,
        .height = (float)context.swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cb, 0, 1, &viewport);

    const VkRect2D scissor = { .extent = context.swapchain_extent };
    vkCmdSetScissor(cb, 0, 1, &scissor);

    const VkDeviceSize vertex_buffer_offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vk_vertex_buffer, &vertex_buffer_offset);
    vkCmdBindIndexBuffer(cb, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const VkDescriptorSet sets[] = {
        vk_descriptor_set_scene,
        vk_descriptor_set_model,
    };
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_layout, 0, 2, sets, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

    vkCmdDrawIndexed(cb, index_count, 1, 0, 0, 0);

    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
}

} // namespace application