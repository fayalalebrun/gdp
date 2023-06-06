#version 310 es

precision mediump float;

layout(location = 0) out vec4 fragColor;

uniform vec3 viewPos;
uniform vec3 color;

in vec3 fFragPos;
in vec3 fNormal;

vec3 calculateDirectionalLight(vec3 normal, vec3 viewDir)
{
  
  vec3 lightDir = normalize(vec3(1.0,1.0,0.0));

  float diff = max(dot(normal, lightDir), 0.0f);

  vec3 halfwayVec = normalize(lightDir + viewDir);


  vec3 ambient = 0.1 * color;
  vec3 diffuse = diff * color;


  return (ambient + diffuse);
}

void main()
{
  vec3 viewDir = normalize(viewPos - fFragPos);
  vec3 normal = normalize(fNormal);
  fragColor = vec4(calculateDirectionalLight(normal, viewDir),1.0);
}
