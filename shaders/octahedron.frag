#version 450 core

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_world_pos;

layout(location = 0) out vec4 out_color;

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

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(std430, set = 0, binding = 1) buffer PointLights {
    PointLight point_lights[];
};

struct SpotLight {
    vec4 position;
    vec4 color;
    vec4 direction;
    float inner_angle;
    float outer_angle;
    float range;
    float _pad;
};

layout(std430, set = 0, binding = 2) buffer SpotLights {
    SpotLight spot_lights[];
};

vec3 blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation) {
    vec3 H = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 diffuse = model_data.albedo * NdotL * light_color;
    vec3 spec = model_data.specular * pow(NdotH, model_data.shininess) * light_color;
    return (diffuse + spec) * attenuation;
}

vec3 shade_directional(vec3 N, vec3 V) {
    vec3 L = normalize(-scene.light_direction.xyz);
    return blinn_phong(N, V, L, scene.light_color.rgb, 1.0);
}

vec3 shade_point(vec3 N, vec3 V, vec3 world_pos, PointLight light) {
    vec3 to_light = light.position.xyz - world_pos;
    float dist = length(to_light);
    vec3 L = to_light / max(dist, 0.0001);
    float attenuation = 1.0 / max(dist * dist, 0.0001);
    return blinn_phong(N, V, L, light.color.rgb, attenuation);
}

vec3 shade_spot(vec3 N, vec3 V, vec3 world_pos, SpotLight light) {
    vec3 to_light = light.position.xyz - world_pos;
    float dist = length(to_light);
    vec3 L = to_light / max(dist, 0.0001);

    vec3 spot_dir = normalize(light.direction.xyz);
    float cos_theta = dot(normalize(-L), spot_dir);
    float spot_factor = smoothstep(light.outer_angle, light.inner_angle, cos_theta);

    float attenuation = 1.0 / max(dist * dist, 0.0001);
    return blinn_phong(N, V, L, light.color.rgb, attenuation * spot_factor);
}

void main() {
    vec3 N = normalize(in_world_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_pos);

    vec3 color = scene.ambient.rgb * model_data.albedo;
    color += shade_directional(N, V);

    for (int i = 0; i < scene.point_light_count; ++i) {
        color += shade_point(N, V, in_world_pos, point_lights[i]);
    }

    for (int i = 0; i < scene.spot_light_count; ++i) {
        color += shade_spot(N, V, in_world_pos, spot_lights[i]);
    }

    out_color = vec4(color, 1.0);
}