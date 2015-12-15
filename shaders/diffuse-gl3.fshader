#version 150

uniform vec3 uLight, uLight2, uColor;
uniform float uTransparency;

in vec3 vNormal;
in vec3 vPosition;

out vec4 fragColor;

void main() {
  vec3 tolight = normalize(uLight - vPosition);
  vec3 tolight2 = normalize(uLight2 - vPosition);
  vec3 normal = normalize(vNormal);

  float diffuse = max(0.0, dot(normal, tolight));
  diffuse += max(0.0, dot(normal, tolight2));
  vec3 intensity = uColor * diffuse;

  if (uColor == vec3(0.6, 0.6, 0.6))
  {
	fragColor = vec4(uColor, 0.3);
  }
  else
  {
	fragColor = vec4(uColor, uTransparency);
  }

  
}
