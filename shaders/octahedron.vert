#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_world_pos;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 light_direction;
    vec4 light_color;
    vec4 ambient;
    int point_light_count;
    int spot_light_count;
    int _pad5;
    int _pad6;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float shininess;
} model_data;

void main() {
    vec4 world_pos = model_data.model * vec4(in_position, 1.0);
    out_world_pos = world_pos.xyz;
    out_world_normal = mat3(model_data.model) * in_normal;
    gl_Position = scene.view_projection * world_pos;
    out_color = in_color;
}