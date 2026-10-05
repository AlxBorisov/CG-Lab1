#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
	mat4 matrix;
	vec4 base_color;
} global_uniforms;

void main() {
	gl_Position = global_uniforms.matrix * vec4(in_position, 1);
	out_color = in_color * global_uniforms.base_color.rgb;
}