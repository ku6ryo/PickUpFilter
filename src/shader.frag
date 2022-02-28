precision mediump float;

uniform sampler2D u_image;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_finger;
uniform float u_openness;

const float E = 8.854187817;

// the texCoords passed in from the vertex shader.
varying vec2 v_texCoord;

void main() {
  float x = v_texCoord.x - (u_finger.x / u_resolution.x);
  float y = v_texCoord.y - (u_finger.y / u_resolution.y);
  float r = sqrt(x*x + y*y);
  float diff = 0.2 * tan((1.0 - 1.33) * cos((r - u_time / 2.0) * 40.0)) * pow(E, -r * 4.0) * (pow(2.0, u_openness) - 1.0);
  float diffX = x / r * diff;
  float diffY = y / r * diff;
  vec4 c = texture2D(u_image, v_texCoord + vec2(diffX, diffY));
  float blueness = clamp(abs(diff) * 20.0, 0.0, 1.0);
  gl_FragColor = vec4(c.rg * vec2(1.0, 1.0) * (1.0 - blueness), c.b, c.a);
}