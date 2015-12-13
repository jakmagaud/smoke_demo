#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <math.h>
#include <time.h>

#if __GNUG__
#   include <tr1/memory>
#endif

#ifdef __MAC__
#   include <OpenGL/gl3.h>
#   include <GLUT/glut.h>
#else
#   include <GL/glew.h>
#   include <GL/glut.h>
#endif

#include "headers/cvec.h"
#include "headers/matrix4.h"
#include "headers/geometrymaker.h"
#include "headers/ppm.h"
#include "headers/glsupport.h"
#include "headers/quat.h"
#include "headers/rigtform.h"
#include "headers/arcball.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
using namespace tr1; // for shared_ptr

#define PI 3.1415926535

// G L O B A L S ///////////////////////////////////////////////////

static const bool g_Gl2Compatible = false;
static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)
static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length
static float g_arcballScreenRadius = 1.0;
static float g_arcballScale = 1.0;
static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false;         // space state, for middle mouse emulation
static bool g_worldFrame = true;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

struct ShaderState {
    GlProgram program;
    
    // Handles to uniform variables
    GLint h_uLight, h_uLight2;
    GLint h_uProjMatrix;
    GLint h_uModelViewMatrix;
    GLint h_uNormalMatrix;
    GLint h_uColor;
    
    // Handles to vertex attributes
    GLint h_aPosition;
    GLint h_aNormal;
    
    ShaderState(const char* vsfn, const char* fsfn) {
        readAndCompileShader(program, vsfn, fsfn);
        
        const GLuint h = program; // short hand
        
        // Retrieve handles to uniform variables
        h_uLight = safe_glGetUniformLocation(h, "uLight");
        h_uLight2 = safe_glGetUniformLocation(h, "uLight2");
        h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
        h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
        h_uNormalMatrix = safe_glGetUniformLocation(h, "uNormalMatrix");
        h_uColor = safe_glGetUniformLocation(h, "uColor");
        
        // Retrieve handles to vertex attributes
        h_aPosition = safe_glGetAttribLocation(h, "aPosition");
        h_aNormal = safe_glGetAttribLocation(h, "aNormal");
        
        if (!g_Gl2Compatible)
            glBindFragDataLocation(h, 0, "fragColor");
        checkGlErrors();
    }
    
};

static const int g_numShaders = 2;
static const char * const g_shaderFiles[g_numShaders][2] = {
    { "./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader" },
    { "./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader" }
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
    { "./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader" },
    { "./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader" }
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
    Cvec3f p, n;
    
    VertexPN() {}
    VertexPN(float x, float y, float z,
             float nx, float ny, float nz)
    : p(x, y, z), n(nx, ny, nz)
    {}
    
    // Define copy constructor and assignment operator from GenericVertex so we can
    // use make* functions from geometrymaker.h
    VertexPN(const GenericVertex& v) {
        *this = v;
    }
    
    VertexPN& operator = (const GenericVertex& v) {
        p = v.pos;
        n = v.normal;
        return *this;
    }
};

struct Geometry {
    GlBufferObject vbo, ibo;
    GlArrayObject vao;
    int vboLen, iboLen;
    
    Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
        this->vboLen = vboLen;
        this->iboLen = iboLen;
        
        // Now create the VBO and IBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
    }
    
    void draw(const ShaderState& curSS) {
        // bind the object's VAO
        glBindVertexArray(vao);
        
        // Enable the attributes used by our shader
        safe_glEnableVertexAttribArray(curSS.h_aPosition);
        safe_glEnableVertexAttribArray(curSS.h_aNormal);
        
        // bind vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
        safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));
        
        // bind ibo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        
        // draw!
        glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);
        
        // Disable the attributes used by our shader
        safe_glDisableVertexAttribArray(curSS.h_aPosition);
        safe_glDisableVertexAttribArray(curSS.h_aNormal);
        
        // disable VAO
        glBindVertexArray(NULL);
    }
};

struct Particle {
    RigTForm rbt;
    Cvec3 velocity; //velocity
    float life;
    shared_ptr<Geometry> sphere;
};

const int MaxParticles = 2000;
Particle particles[MaxParticles];
const int ParticleRadius = 0.2;

// Vertex buffer and index buffer associated with the ground and cube geometry and sphere
static shared_ptr<Geometry> g_ground, g_sphere;

// --------- Scene
static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
static RigTForm g_skyRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
RigTForm eyeRbt;
static RigTForm g_sphereRbt = RigTForm(Cvec3(0.0, 0.0, 0.0));
Cvec3 g_sphereEyeCoord;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void initGround() {
    // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
    VertexPN vtx[4] = {
        VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
        VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
        VertexPN(g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
        VertexPN(g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    };
    unsigned short idx[] = { 0, 1, 2, 0, 2, 3 };
    g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initParticles() {
    for (int i = 0; i < MaxParticles; i++) {
        //physics
        particles[i].rbt = RigTForm(Cvec3(0.0, 0.0, 0.0));
        particles[i].life = 5;
        
        float spread = .05 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.5)));
        /*Cvec3 gen_dir = Cvec3(0.0, 1, 0.0); //general direction of the particles
        Cvec3 rand_dir = Cvec3((rand()%2000 - 1000.0)/1000.0,(rand()%2000)/1000.0, 0); //generate a random component for each one
        //particles[i].velocity = Cvec3(gen_dir[0] + rand_dir * spread, gen_dir[1] + rand_dir * spread, gen_dir[2] + rand_dir * spread);
        particles[i].velocity = /*Cvec3(0.01 * i,0.01 * i,0) + rand_dir;*/
        float angle = PI/3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(PI/3)));
        particles[i].velocity = Cvec3(cos(angle), sin(angle), 0) * spread;
        
        //geometry
        int ibLen, vbLen;
        getSphereVbIbLen(5, 5, vbLen, ibLen);
        
        vector<VertexPN> vtx(vbLen);
        vector<unsigned short> idx(ibLen);
        makeSphere(1, 5, 5, vtx.begin(), idx.begin());
        particles[i].sphere.reset(new Geometry(&vtx[0], &idx[0], vtx.size(), idx.size()));
        
    }
}

static void initSphere() {
    int ibLen, vbLen;
    getSphereVbIbLen(20, 10, vbLen, ibLen);
    
    //temporary storage for sphere geometry
    vector<VertexPN> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    
    makeSphere(1.0, 20, 10, vtx.begin(), idx.begin());
    g_sphere.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
    GLfloat glmatrix[16];
    projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
    safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// takes MVM and its normal matrix to the shaders
static void sendModelViewNormalMatrix(const ShaderState& curSS, const Matrix4& MVM, const Matrix4& NMVM) {
    GLfloat glmatrix[16];
    MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
    safe_glUniformMatrix4fv(curSS.h_uModelViewMatrix, glmatrix);
    
    NMVM.writeToColumnMajorMatrix(glmatrix); // send NMVM
    safe_glUniformMatrix4fv(curSS.h_uNormalMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
    if (g_windowWidth >= g_windowHeight)
        g_frustFovY = g_frustMinFov;
    else {
        const double RAD_PER_DEG = 0.5 * CS175_PI / 180;
        g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
    }
}

static Matrix4 makeProjectionMatrix() {
    return Matrix4::makeProjection(g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),g_frustNear, g_frustFar);
}

static void drawStuff() {
    //get eye coordinates of the center of the sphere
    g_sphereEyeCoord = Cvec3(inv(eyeRbt) * Cvec4(g_sphereRbt.getTranslation(), 1.0));
    if (g_mouseLClickButton == false && g_mouseRClickButton == false)
    {
        g_arcballScale = getScreenToEyeScale(g_sphereEyeCoord[2], g_frustFovY, g_windowHeight);
    }
    
    // short hand for current shader state
    const ShaderState& curSS = *g_shaderStates[g_activeShader];
    
    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    sendProjectionMatrix(curSS, projmat);
    
    eyeRbt = g_skyRbt;
    const RigTForm invEyeRbt = inv(eyeRbt);
    
    const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
    const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
    safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
    safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);
    
    // draw ground
    // ===========
    //
    const RigTForm groundRbt = RigTForm();  // identity
    Matrix4 MVM = rigTFormToMatrix(invEyeRbt * groundRbt);
    Matrix4 NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, 0.1, 0.95, 0.1); // set color
    g_ground->draw(curSS);
    
    // draw sphere
    // ===========
    /*
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    MVM = rigTFormToMatrix(invEyeRbt * g_sphereRbt) * Matrix4::makeScale(Cvec3 (g_arcballScale * g_arcballScreenRadius, g_arcballScale * g_arcballScreenRadius, g_arcballScale * g_arcballScreenRadius));
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, 0.0, 0.8, 0.0); // set color
    if (g_worldFrame == true)
    {
        g_sphere->draw(curSS);
    }
    */
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    
    for (int i = 0; i < MaxParticles; i++) {
        //RigTForm sphere = inv(particles[i].rbt);
        //cout << sphere.getTranslation()[0] << " " << sphere.getTranslation()[1] << "\n";
        Cvec3 newpos = particles[i].rbt.getTranslation() + particles[i].velocity;
        if (newpos[0] > 2.75 || newpos[0] < -2.75|| newpos[1] > 2.75)
            newpos = Cvec3(0,0,0);
        particles[i].rbt.setTranslation(newpos);
        //cout << newpos[0] << " " << newpos[1] << "\n";
        //cout << sphere.getTranslation()[0] << " " << sphere.getTranslation()[1] << "\n";
        //cout << particles[i].velocity[0] << " " << particles[i].velocity[1] << "\n";
        Matrix4 MVM = rigTFormToMatrix(invEyeRbt * particles[i].rbt) * Matrix4::makeScale(Cvec3(0.02, 0.02, 0.02));
        sendModelViewNormalMatrix(curSS, MVM, normalMatrix(MVM));
        safe_glUniform3f(curSS.h_uColor, 0.69, 0.69, 0.69); // set color to grayish
        particles[i].sphere->draw(curSS);
    }
    glutPostRedisplay();
}

static void display() {
    glUseProgram(g_shaderStates[g_activeShader]->program);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth
    
    drawStuff();
    
    glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)
    
    checkGlErrors();
}

static void reshape(const int w, const int h) {
    g_windowWidth = w;
    g_windowHeight = h;
    glViewport(0, 0, w, h);
    g_arcballScreenRadius = 0.25 * min(g_windowWidth, g_windowHeight);
    
    cerr << "Size of window is now " << w << "x" << h << endl;
    updateFrustFovY();
    glutPostRedisplay();
}

static void motion(const int x, const int y) {
    
    glutPostRedisplay(); // we always redraw if we changed the scene
}

static void mouse(const int button, const int state, const int x, const int y) {
    g_mouseClickX = x;
    g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system
    
    g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
    g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
    g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);
    
    g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
    g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
    g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);
    
    g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
    
    glutPostRedisplay();
}

static void keyboardUp(const unsigned char key, const int x, const int y) {
    switch (key) {
        case ' ':
            g_spaceDown = false;
            break;
    }
    glutPostRedisplay();
}

static void keyboard(const unsigned char key, const int x, const int y) {
    switch (key) {
        case 27:
            exit(0);                                  // ESC
        case 'h':
            cout << " ============== H E L P ==============\n\n"
            << "h\t\thelp menu\n"
            << "s\t\tsave screenshot\n"
            << "f\t\tToggle flat shading on/off.\n"
            << "o\t\tCycle object to edit\n"
            << "v\t\tCycle view\n"
            << "m\t\Cycles through world-sky and sky-sky frames\n"
            << "drag left mouse to rotate\n"
            << "drag right mouse to translate\n" << endl;
            break;
        case 's':
            glFlush();
            writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
            break;
        case 'f':
            g_activeShader ^= 1;
            break;
        case ' ':
            g_spaceDown = true;
            break;
            
    }
    glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
    glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
#ifdef __MAC__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH); // core profile flag is required for GL 3.2 on Mac
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);  //  RGBA pixel channels and double buffering
#endif
    glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
    glutCreateWindow("Final Project");                       // title the window
    
    glutIgnoreKeyRepeat(true);                              // avoids repeated keyboard calls when holding space to emulate middle mouse
    
    glutDisplayFunc(display);                               // display rendering callback
    glutReshapeFunc(reshape);                               // window reshape callback
    glutMotionFunc(motion);                                 // mouse movement callback
    glutMouseFunc(mouse);                                   // mouse click callback
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardUp);
}

static void initGLState() {
    glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);
    if (!g_Gl2Compatible)
        glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
    g_shaderStates.resize(g_numShaders);
    for (int i = 0; i < g_numShaders; ++i) {
        if (g_Gl2Compatible)
            g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
        else
            g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
    }
}

static void initGeometry() {
    initGround();
    initSphere();
    initParticles();
}

int main(int argc, char * argv[]) {
    try {
        initGlutState(argc, argv);
        
        // on Mac, we shouldn't use GLEW.
        
#ifndef __MAC__
        glewInit(); // load the OpenGL extensions
#endif
        
        cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.5") << endl;
        
#ifndef __MAC__
        if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
            throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
        else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
            throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");
#endif
        
        initGLState();
        initShaders();
        initGeometry();
        
        glutMainLoop();
        return 0;
    }
    catch (const runtime_error& e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
