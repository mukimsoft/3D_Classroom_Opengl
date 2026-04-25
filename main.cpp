#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

struct Vec3 {
    float x, y, z;
};

struct Mat4 {
    float m[16]{};
};

static constexpr float PI = 3.1415926535f;

static Mat4 identity() {
    Mat4 r;
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    if (len <= 0.00001f) return { 0, 0, 0 };
    return { v.x / len, v.y / len, v.z / len };
}

static Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            r.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }
    return r;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 scale(float x, float y, float z) {
    Mat4 r{};
    r.m[0] = x;
    r.m[5] = y;
    r.m[10] = z;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 rotateX(float deg) {
    float a = deg * PI / 180.0f;
    float c = std::cos(a);
    float s = std::sin(a);
    Mat4 r = identity();
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 rotateY(float deg) {
    float a = deg * PI / 180.0f;
    float c = std::cos(a);
    float s = std::sin(a);
    Mat4 r = identity();
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

static Mat4 rotateZ(float deg) {
    float a = deg * PI / 180.0f;
    float c = std::cos(a);
    float s = std::sin(a);
    Mat4 r = identity();
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}

static Mat4 perspective(float fovDeg, float aspect, float nearPlane, float farPlane) {
    float f = 1.0f / std::tan(fovDeg * PI / 360.0f);
    Mat4 r{};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    return r;
}

static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize({ center.x - eye.x, center.y - eye.y, center.z - eye.z });
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;

    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log.data() << std::endl;
        std::exit(1);
    }
    return shader;
}

static GLuint createProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetProgramInfoLog(program, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log.data() << std::endl;
        std::exit(1);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

static GLuint createTexture1x1(unsigned char r, unsigned char g, unsigned char b) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    unsigned char data[3] = { r, g, b };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return tex;
}

static GLuint createCheckerTexture(int width, int height,
    unsigned char r1, unsigned char g1, unsigned char b1,
    unsigned char r2, unsigned char g2, unsigned char b2) {
    std::vector<unsigned char> pixels(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool check = ((x / 16) + (y / 16)) % 2 == 0;
            int idx = (y * width + x) * 3;
            pixels[idx + 0] = check ? r1 : r2;
            pixels[idx + 1] = check ? g1 : g2;
            pixels[idx + 2] = check ? b1 : b2;
        }
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    return tex;
}

static GLuint createStripeTexture(int width, int height,
    unsigned char r1, unsigned char g1, unsigned char b1,
    unsigned char r2, unsigned char g2, unsigned char b2) {
    std::vector<unsigned char> pixels(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool stripe = ((x / 8) % 2) == 0;
            int idx = (y * width + x) * 3;
            pixels[idx + 0] = stripe ? r1 : r2;
            pixels[idx + 1] = stripe ? g1 : g2;
            pixels[idx + 2] = stripe ? b1 : b2;
        }
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    return tex;
}

// Better bitmap font - 8x12 pixels per character
static unsigned char getCharPixel(char c, int x, int y) {
    // Simple 8x12 font patterns for common characters
    static const unsigned char fontData[][12] = {
        // A
        {0x00,0x18,0x24,0x24,0x3C,0x42,0x42,0x42,0x42,0x42,0x00,0x00},
        // B
        {0x00,0x3C,0x42,0x42,0x3C,0x42,0x42,0x42,0x42,0x3C,0x00,0x00},
        // C
        {0x00,0x18,0x24,0x40,0x40,0x40,0x40,0x40,0x24,0x18,0x00,0x00},
        // D
        {0x00,0x38,0x44,0x42,0x42,0x42,0x42,0x42,0x44,0x38,0x00,0x00},
        // E
        {0x00,0x7C,0x40,0x40,0x78,0x40,0x40,0x40,0x40,0x7C,0x00,0x00},
        // F
        {0x00,0x7C,0x40,0x40,0x78,0x40,0x40,0x40,0x40,0x40,0x00,0x00},
        // G
        {0x00,0x18,0x24,0x40,0x40,0x40,0x5C,0x44,0x24,0x18,0x00,0x00},
        // H
        {0x00,0x42,0x42,0x42,0x42,0x7E,0x42,0x42,0x42,0x42,0x00,0x00},
        // I
        {0x00,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00},
        // J
        {0x00,0x1E,0x08,0x08,0x08,0x08,0x08,0x48,0x48,0x30,0x00,0x00},
        // K
        {0x00,0x42,0x44,0x48,0x50,0x60,0x50,0x48,0x44,0x42,0x00,0x00},
        // L
        {0x00,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x7C,0x00,0x00},
        // M
        {0x00,0x42,0x66,0x66,0x5A,0x5A,0x42,0x42,0x42,0x42,0x00,0x00},
        // N
        {0x00,0x42,0x62,0x52,0x4A,0x46,0x42,0x42,0x42,0x42,0x00,0x00},
        // O
        {0x00,0x18,0x24,0x42,0x42,0x42,0x42,0x42,0x24,0x18,0x00,0x00},
        // P
        {0x00,0x3C,0x42,0x42,0x42,0x3C,0x40,0x40,0x40,0x40,0x00,0x00},
        // Q
        {0x00,0x18,0x24,0x42,0x42,0x42,0x42,0x4A,0x24,0x1A,0x00,0x00},
        // R
        {0x00,0x3C,0x42,0x42,0x42,0x3C,0x48,0x44,0x42,0x42,0x00,0x00},
        // S
        {0x00,0x1C,0x22,0x20,0x1C,0x02,0x02,0x22,0x22,0x1C,0x00,0x00},
        // T
        {0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00},
        // U
        {0x00,0x42,0x42,0x42,0x42,0x42,0x42,0x42,0x24,0x18,0x00,0x00},
        // V
        {0x00,0x42,0x42,0x42,0x42,0x42,0x24,0x24,0x18,0x18,0x00,0x00},
        // W
        {0x00,0x42,0x42,0x42,0x42,0x5A,0x5A,0x66,0x66,0x42,0x00,0x00},
        // X
        {0x00,0x42,0x42,0x24,0x24,0x18,0x18,0x24,0x24,0x42,0x00,0x00},
        // Y
        {0x00,0x42,0x42,0x24,0x24,0x18,0x18,0x18,0x18,0x18,0x00,0x00},
        // Z
        {0x00,0x7C,0x02,0x04,0x08,0x10,0x20,0x40,0x40,0x7C,0x00,0x00},
        // 0
        {0x00,0x18,0x24,0x42,0x46,0x4A,0x52,0x62,0x42,0x3C,0x00,0x00},
        // 1
        {0x00,0x18,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00},
        // 2
        {0x00,0x3C,0x42,0x02,0x04,0x08,0x10,0x20,0x40,0x7E,0x00,0x00},
        // 3
        {0x00,0x3C,0x42,0x02,0x04,0x18,0x04,0x02,0x42,0x3C,0x00,0x00},
        // 4
        {0x00,0x04,0x0C,0x14,0x24,0x44,0x7E,0x04,0x04,0x0E,0x00,0x00},
        // 5
        {0x00,0x7E,0x40,0x40,0x7C,0x02,0x02,0x02,0x42,0x3C,0x00,0x00},
        // 6
        {0x00,0x1C,0x20,0x40,0x7C,0x42,0x42,0x42,0x42,0x3C,0x00,0x00},
        // 7
        {0x00,0x7E,0x02,0x04,0x08,0x10,0x10,0x10,0x10,0x10,0x00,0x00},
        // 8
        {0x00,0x3C,0x42,0x42,0x42,0x3C,0x42,0x42,0x42,0x3C,0x00,0x00},
        // 9
        {0x00,0x3C,0x42,0x42,0x42,0x3E,0x02,0x02,0x04,0x38,0x00,0x00},
        // Space
        {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
        // :
        {0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00,0x00,0x00},
        // -
        {0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00},
    };

    int idx = 0;
    if (c >= 'A' && c <= 'Z') idx = c - 'A';
    else if (c >= '0' && c <= '9') idx = 26 + (c - '0');
    else if (c == ' ') idx = 36;
    else if (c == ':') idx = 37;
    else if (c == '-') idx = 38;
    else return 0;

    if (x < 0 || x >= 8 || y < 0 || y >= 12) return 0;
    return (fontData[idx][y] >> (7 - x)) & 1;
}

static GLuint createTextTexture(const char* text, int texWidth, int texHeight,
    unsigned char fgR, unsigned char fgG, unsigned char fgB) {

    std::vector<unsigned char> pixels(texWidth * texHeight * 3, 255);

    // Fill with white background
    for (int i = 0; i < texWidth * texHeight; ++i) {
        pixels[i * 3 + 0] = 255;
        pixels[i * 3 + 1] = 255;
        pixels[i * 3 + 2] = 255;
    }

    // Render text - start from top-left
    int startX = 10;
    int startY = texHeight - 5;
    int charWidth = 8;
    int charHeight = 12;
    int spacing = 2;

    int x = startX;
    for (const char* p = text; *p && x < texWidth - 10; ++p) {
        if (*p == ' ') {
            x += charWidth + spacing;
            continue;
        }

        // Render character
        for (int cy = 0; cy < charHeight && (startY - cy) >= 0; ++cy) {
            for (int cx = 0; cx < charWidth; ++cx) {
                if (getCharPixel(*p, cx, cy)) {
                    int px = x + cx;
                    int py = startY - cy;
                    if (px < texWidth && py >= 0 && py < texHeight) {
                        int idx = (py * texWidth + px) * 3;
                        pixels[idx + 0] = fgR;
                        pixels[idx + 1] = fgG;
                        pixels[idx + 2] = fgB;
                    }
                }
            }
        }
        x += charWidth + spacing;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static GLuint cubeVAO = 0;
static GLuint cubeVBO = 0;

static void initCubeMesh() {
    float vertices[] = {
        // positions            // normals         // texcoords
        // Front
        -0.5f, -0.5f,  0.5f,    0, 0, 1,          0, 0,
         0.5f, -0.5f,  0.5f,    0, 0, 1,          1, 0,
         0.5f,  0.5f,  0.5f,    0, 0, 1,          1, 1,
         0.5f,  0.5f,  0.5f,    0, 0, 1,          1, 1,
        -0.5f,  0.5f,  0.5f,    0, 0, 1,          0, 1,
        -0.5f, -0.5f,  0.5f,    0, 0, 1,          0, 0,

        // Back
        -0.5f, -0.5f, -0.5f,    0, 0, -1,         1, 0,
        -0.5f,  0.5f, -0.5f,    0, 0, -1,         1, 1,
         0.5f,  0.5f, -0.5f,    0, 0, -1,         0, 1,
         0.5f,  0.5f, -0.5f,    0, 0, -1,         0, 1,
         0.5f, -0.5f, -0.5f,    0, 0, -1,         0, 0,
        -0.5f, -0.5f, -0.5f,    0, 0, -1,         1, 0,

        // Left
        -0.5f, -0.5f, -0.5f,   -1, 0, 0,          0, 0,
        -0.5f, -0.5f,  0.5f,   -1, 0, 0,          1, 0,
        -0.5f,  0.5f,  0.5f,   -1, 0, 0,          1, 1,
        -0.5f,  0.5f,  0.5f,   -1, 0, 0,          1, 1,
        -0.5f,  0.5f, -0.5f,   -1, 0, 0,          0, 1,
        -0.5f, -0.5f, -0.5f,   -1, 0, 0,          0, 0,

        // Right
         0.5f, -0.5f, -0.5f,    1, 0, 0,          1, 0,
         0.5f,  0.5f,  0.5f,    1, 0, 0,          0, 1,
         0.5f, -0.5f,  0.5f,    1, 0, 0,          0, 0,
         0.5f,  0.5f,  0.5f,    1, 0, 0,          0, 1,
         0.5f, -0.5f, -0.5f,    1, 0, 0,          1, 0,
         0.5f,  0.5f, -0.5f,    1, 0, 0,          1, 1,

         // Top
         -0.5f,  0.5f, -0.5f,    0, 1, 0,          0, 1,
         -0.5f,  0.5f,  0.5f,    0, 1, 0,          0, 0,
          0.5f,  0.5f,  0.5f,    0, 1, 0,          1, 0,
          0.5f,  0.5f,  0.5f,    0, 1, 0,          1, 0,
          0.5f,  0.5f, -0.5f,    0, 1, 0,          1, 1,
         -0.5f,  0.5f, -0.5f,    0, 1, 0,          0, 1,

         // Bottom
         -0.5f, -0.5f, -0.5f,    0, -1, 0,         0, 0,
          0.5f, -0.5f,  0.5f,    0, -1, 0,         1, 1,
         -0.5f, -0.5f,  0.5f,    0, -1, 0,         0, 1,
          0.5f, -0.5f,  0.5f,    0, -1, 0,         1, 1,
         -0.5f, -0.5f, -0.5f,    0, -1, 0,         0, 0,
          0.5f, -0.5f, -0.5f,    0, -1, 0,         1, 0
    };

    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);

    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

static void drawCube(GLuint program, const Mat4& model, GLuint texture) {
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE, model.m);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

static void drawBox(GLuint program, GLuint texture, float x, float y, float z, float sx, float sy, float sz, float rotY = 0.0f) {
    Mat4 model = multiply(translate(x, y, z), multiply(rotateY(rotY), scale(sx, sy, sz)));
    drawCube(program, model, texture);
}

static void drawCeilingFan(GLuint program, GLuint metalTex, GLuint bladeTex, float x, float y, float z, float time) {
    drawBox(program, metalTex, x, y, z, 0.15f, 0.08f, 0.15f);

    float bladeLen = 0.6f;
    float bladeW = 0.04f;
    float bladeH = 0.02f;
    float angleOffset = time * 2.0f;

    for (int i = 0; i < 4; ++i) {
        float angle = angleOffset + i * 90.0f;
        Mat4 bladeModel = multiply(translate(x, y - 0.03f, z),
            multiply(rotateY(angle),
                multiply(rotateZ(5.0f),
                    scale(bladeLen, bladeH, bladeW))));
        drawCube(program, bladeModel, bladeTex);
    }

    drawBox(program, metalTex, x, y - 0.1f, z, 0.12f, 0.04f, 0.12f);
}

static void drawCeilingLight(GLuint program, GLuint lightTex, float x, float y, float z) {
    drawBox(program, lightTex, x, y, z, 0.3f, 0.08f, 0.3f);
    drawBox(program, lightTex, x, y - 0.06f, z, 0.15f, 0.12f, 0.15f);
}

static void drawDesk(GLuint program, GLuint woodTex, float x, float y, float z, float w, float h, float d) {
    float legW = w * 0.08f;
    float legD = d * 0.08f;
    float topH = h * 0.12f;

    drawBox(program, woodTex, x, y + h * 0.95f, z, w, topH, d);

    float legH = h * 0.95f;
    drawBox(program, woodTex, x - w * 0.42f, y + legH * 0.5f, z - d * 0.38f, legW, legH, legD);
    drawBox(program, woodTex, x + w * 0.42f, y + legH * 0.5f, z - d * 0.38f, legW, legH, legD);
    drawBox(program, woodTex, x - w * 0.42f, y + legH * 0.5f, z + d * 0.38f, legW, legH, legD);
    drawBox(program, woodTex, x + w * 0.42f, y + legH * 0.5f, z + d * 0.38f, legW, legH, legD);
}

static void drawChair(GLuint program, GLuint woodTex, float x, float y, float z) {
    float seatW = 0.8f;
    float seatH = 0.08f;
    float seatD = 0.8f;

    float legW = 0.06f;
    float legD = 0.06f;
    float legH = 0.45f;

    drawBox(program, woodTex, x, y + legH, z, seatW, seatH, seatD);

    drawBox(program, woodTex, x - 0.33f, y + legH * 0.5f, z - 0.33f, legW, legH, legD);
    drawBox(program, woodTex, x + 0.33f, y + legH * 0.5f, z - 0.33f, legW, legH, legD);
    drawBox(program, woodTex, x - 0.33f, y + legH * 0.5f, z + 0.33f, legW, legH, legD);
    drawBox(program, woodTex, x + 0.33f, y + legH * 0.5f, z + 0.33f, legW, legH, legD);

    drawBox(program, woodTex, x, y + legH + 0.5f, z + 0.32f, 0.7f, 0.7f, 0.08f);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "3D Classroom - CSE 4288", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return 1;
    }

    glEnable(GL_DEPTH_TEST);

    const char* vertexShaderSrc = R"GLSL(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;

        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;

        void main() {
            FragPos = vec3(uModel * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(uModel))) * aNormal;
            TexCoord = aTexCoord;
            gl_Position = uProjection * uView * vec4(FragPos, 1.0);
        }
    )GLSL";

    const char* fragmentShaderSrc = R"GLSL(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;

        out vec4 FragColor;

        uniform sampler2D uTexture;
        uniform vec3 uLightDir;
        uniform vec3 uAmbientColor;
        uniform vec3 uDiffuseColor;
        uniform vec3 uSpecColor;
        uniform vec3 uViewPos;

        void main() {
            vec3 tex = texture(uTexture, TexCoord).rgb;
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(-uLightDir);

            float diff = max(dot(norm, lightDir), 0.0);

            vec3 viewDir = normalize(uViewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);

            vec3 ambient = uAmbientColor * tex;
            vec3 diffuse = uDiffuseColor * diff * tex;
            vec3 specular = uSpecColor * spec;

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0);
        }
    )GLSL";

    GLuint program = createProgram(vertexShaderSrc, fragmentShaderSrc);
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTexture"), 0);

    initCubeMesh();

    // === TEXTURES ===
    GLuint furnitureTex = createStripeTexture(128, 128, 180, 140, 90, 140, 100, 60);
    GLuint wallTex = createTexture1x1(255, 255, 255);
    GLuint floorTex = createCheckerTexture(128, 128, 190, 180, 165, 170, 160, 145);
    GLuint acTex = createStripeTexture(128, 128, 190, 195, 200, 160, 165, 170);
    GLuint projectorTex = createTexture1x1(45, 45, 50);
    GLuint lightTex = createTexture1x1(255, 250, 200);
    GLuint fanMetalTex = createTexture1x1(180, 180, 190);
    GLuint fanBladeTex = createTexture1x1(220, 220, 230);
    GLuint samsungTex = createTextTexture("SAMSUNG", 128, 32, 10, 10, 150);

    // IMPROVED: Clear, bold text with high contrast
    GLuint headerTex = createTextTexture("CSE 4288: Computer Graphics Lab", 640, 80, 0, 0, 100);
    GLuint lecturerTex = createTextTexture("Lecturer : Maimuna Chowdhury Disha", 640, 80, 0, 80, 0);
    GLuint groupTex = createTextTexture("Project Group ID : 405, 332, 336, 310", 640, 80, 100, 0, 0);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 16.0f / 9.0f;

        glClearColor(0.78f, 0.86f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Mat4 view = lookAt({ 0.0f, 2.8f, 10.5f }, { 0.0f, 1.8f, -1.5f }, { 0.0f, 1.0f, 0.0f });
        Mat4 proj = perspective(45.0f, aspect, 0.1f, 100.0f);

        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "uView"), 1, GL_FALSE, view.m);
        glUniformMatrix4fv(glGetUniformLocation(program, "uProjection"), 1, GL_FALSE, proj.m);

        glUniform3f(glGetUniformLocation(program, "uLightDir"), -0.5f, -1.0f, -0.3f);
        glUniform3f(glGetUniformLocation(program, "uAmbientColor"), 0.35f, 0.35f, 0.38f);
        glUniform3f(glGetUniformLocation(program, "uDiffuseColor"), 0.95f, 0.95f, 0.95f);
        glUniform3f(glGetUniformLocation(program, "uSpecColor"), 0.22f, 0.22f, 0.22f);
        glUniform3f(glGetUniformLocation(program, "uViewPos"), 0.0f, 2.8f, 10.5f);

        double currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // === ROOM ===
        drawBox(program, floorTex, 0.0f, -0.05f, 0.0f, 12.0f, 0.1f, 9.0f);
        drawBox(program, wallTex, 0.0f, 2.5f, -4.5f, 12.0f, 5.0f, 0.1f);
        drawBox(program, wallTex, -6.0f, 2.5f, 0.0f, 0.1f, 5.0f, 9.0f);
        drawBox(program, wallTex, 6.0f, 2.5f, 0.0f, 0.1f, 5.0f, 9.0f);
        drawBox(program, wallTex, 0.0f, 5.0f, 0.0f, 12.0f, 0.1f, 9.0f);

        // === WHITEBOARD with CLEAR TEXT ===
        // Dark brown frame
        drawBox(program, createTexture1x1(101, 67, 33), 0.0f, 2.55f, -4.35f, 4.9f, 2.05f, 0.12f);
        // White board surface
        drawBox(program, createTexture1x1(255, 255, 255), 0.0f, 2.55f, -4.27f, 4.5f, 1.8f, 0.04f);

        // Text lines - positioned with proper spacing and larger size
        // Header - Top line (Dark Blue)
        drawBox(program, headerTex, 0.0f, 3.9f, -4.25f, 4.3f, 0.45f, 0.02f);

        // Lecturer - Middle line (Dark Green)
        drawBox(program, lecturerTex, 0.0f, 3.2f, -4.25f, 4.3f, 0.45f, 0.02f);

        // Group ID - Bottom line (Dark Red)
        drawBox(program, groupTex, 0.0f, 2.5f, -4.25f, 4.3f, 0.45f, 0.02f);

        // === TEACHER TABLE ===
        drawDesk(program, furnitureTex, 0.0f, 0.0f, -2.9f, 2.4f, 0.85f, 0.9f);

        // === PROJECTOR ===
        drawBox(program, projectorTex, 0.0f, 4.35f, -0.6f, 0.8f, 0.2f, 0.5f);
        drawBox(program, projectorTex, 0.0f, 4.65f, -0.6f, 0.08f, 0.6f, 0.08f);

        // === AC UNIT with SAMSUNG ===
        drawBox(program, acTex, -5.75f, 3.85f, 2.1f, 1.2f, 0.7f, 0.5f);
        drawBox(program, wallTex, -5.75f, 3.65f, 2.1f, 1.0f, 0.05f, 0.45f);
        drawBox(program, samsungTex, -5.75f, 4.0f, 2.35f, 0.6f, 0.15f, 0.02f);

        // === CEILING LIGHTS ===
        drawCeilingLight(program, lightTex, -4.0f, 4.95f, 0.0f);
        drawCeilingLight(program, lightTex, 0.0f, 4.95f, 0.0f);
        drawCeilingLight(program, lightTex, 4.0f, 4.95f, 0.0f);
        drawCeilingLight(program, lightTex, -2.0f, 4.95f, -2.0f);
        drawCeilingLight(program, lightTex, 2.0f, 4.95f, -2.0f);

        // === CEILING FANS ===
        drawCeilingFan(program, fanMetalTex, fanBladeTex, -3.0f, 4.85f, 3.0f, currentTime);
        drawCeilingFan(program, fanMetalTex, fanBladeTex, 3.0f, 4.85f, 3.0f, currentTime);

        // === STUDENT DESKS + CHAIRS ===
        const float xs[3] = { -3.4f, 0.0f, 3.4f };
        const float zs[2] = { 0.8f, 2.9f };

        for (float z : zs) {
            for (float x : xs) {
                drawDesk(program, furnitureTex, x, 0.0f, z, 1.25f, 0.75f, 0.75f);
                drawChair(program, furnitureTex, x, 0.0f, z + 1.0f);
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(program);
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
