#version 310 es

precision mediump float;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 fFragPos;
out vec3 fNormal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
  mat4 mvp = projection * view * model;
  fFragPos = vec3(mvp * vec4(position, 1.0f));
  fNormal = mat3(transpose(inverse(model))) * normal;
  
  gl_Position = mvp * vec4(position, 1.0f);
}
