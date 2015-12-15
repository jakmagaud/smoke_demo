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
#   include "GL/glew.h"
#   include "GL/glut.h"
#endif

#include "cvec.h"
#include "matrix4.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "quat.h"
#include "rigtform.h"
#include "arcball.h"

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
const int MaxParticles = 3000;

struct ShaderState {
	GlProgram program;

	// Handles to uniform variables
	GLint h_uLight, h_uLight2;
	GLint h_uProjMatrix;
	GLint h_uModelViewMatrix;
	GLint h_uNormalMatrix;
	GLint h_uColor;
	GLint h_uTransparency;

	// Handles to vertex attributes
	GLint h_aPosition;
	GLint h_aNormal;

	ShaderState(const char* vsfn, const char* fsfn) {
		readAndCompileShader(program, vsfn, fsfn);

		const GLuint h = program; // short hand

								  // Retrieve handles to uniform variables
		h_uLight = safe_glGetUniformLocation(h, "uLight");
		h_uLight2 = safe_glGetUniformLocation(h, "uLight2");
		h_uTransparency = safe_glGetUniformLocation(h, "uTransparency");
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
	Cvec3 color;    //color
	Cvec3 gravitational_force;
	float life;
	float age;
	float scale;
	int type; //0 or 1, either fire(0) or smoke(1)
	shared_ptr<Geometry> sphere;
};

Particle particles[MaxParticles];



// Vertex buffer and index buffer associated with the ground and cube geometry and sphere
static shared_ptr<Geometry> g_ground, g_sphere;

// --------- Scene
static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
static RigTForm g_skyRbt = RigTForm(Cvec3(0.0, 3.0, 20.0));
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

void initParticleAttributes(Particle *particles)
{
	if (particles->type == 0)
	{
		particles->rbt = RigTForm(Cvec3(((rand() % 2) - (rand() % 2)), -5.0, 0.0));
		particles->life = (((rand() % 10 + 1))) / 10.0;
		particles->age = 0.0;
		particles->type = 0;

		particles->velocity[0] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007) - (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007);
		particles->velocity[1] = ((((((5) * rand() % 11) + 5)) * rand() % 11) + 1) * 0.02;
		particles->velocity[2] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007) - (((((((2) * rand() % 11) + 1)) * rand() % 5) + 1) * 0.007);

		particles->color = Cvec3(1.0, 0.95, 0.8);

		particles->gravitational_force = Cvec3(0.0, 0.0, 0.0);
	}
	else
	{
		particles->rbt = RigTForm(Cvec3(((rand() % 2) - (rand() % 2)), -5.0, 0.0));
		particles->life = (((rand() % 10 + 1))) / 10.0;
		particles->age = 0.0;

		particles->velocity[0] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007) - (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007);
		particles->velocity[1] = ((((((5) * rand() % 11) + 5)) * rand() % 11) + 1) * 0.02;
		particles->velocity[2] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.007) - (((((((2) * rand() % 11) + 1)) * rand() % 5) + 1) * 0.007);

		particles->color = Cvec3(1.0, 0.95, 0.8);
	}


}

static void initParticles() {
	for (int i = 0; i < MaxParticles; i++) {
		//physics
		initParticleAttributes(&particles[i]);

		//geometry
		int ibLen, vbLen;
		getSphereVbIbLen(5, 5, vbLen, ibLen);

		vector<VertexPN> vtx(vbLen);
		vector<unsigned short> idx(ibLen);
		makeSphere(10, 5, 5, vtx.begin(), idx.begin());
		particles[i].sphere.reset(new Geometry(&vtx[0], &idx[0], vtx.size(), idx.size()));

	}
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
	return Matrix4::makeProjection(g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight), g_frustNear, g_frustFar);
}

void Smoke_conversion(Particle *particles)
{
	particles->life = (((rand() % 125 + 1) / 10.0) + 5);
	particles->age = 0.0;
	particles->type = 1;

	particles->velocity[0] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.0035) - (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.0035);
	particles->velocity[1] = ((((((5) * rand() % 11) + 3)) * rand() % 11) + 7) * 0.015;
	particles->velocity[2] = (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.0015) - (((((((2) * rand() % 11) + 1)) * rand() % 11) + 1) * 0.0015);

	particles->color = Cvec3(0.6, 0.6, 0.6);
}

void updateParticles()
{
	for (int i = 0; i < MaxParticles; i++)
	{
		//update age
		particles[i].age = particles[i].age + 0.02;

		//update position
		particles[i].rbt.setTranslation(particles[i].rbt.getTranslation() + particles[i].velocity + particles[i].gravitational_force);

		//update gravitational force

		if (particles[i].type == 0)
		{
			particles[i].gravitational_force[1] += 0.005;
		}
		else
		{
			particles[i].gravitational_force[1] += 0.0005;
		}


		//update color
		if (particles[i].type == 0)
		{
			float prob = particles[i].life / particles[i].age;
			if (prob < 1.75)
			{//red
				particles[i].color = Cvec3(1.0, 0.2, 0.0);
			}
			else if (prob < 3.0)
			{//gold
				particles[i].color = Cvec3(1.0, 0.8, 0.0);
			}
			else if (prob < 10.0)
			{//yellow
				particles[i].color = Cvec3(1.0, 1.0, 0.0);
			}
			else
			{// initial light yellow
				particles[i].color = Cvec3(1.0, 0.95, 0.8);
			}
		}


		//update "dead or alive" status of particles
		if (particles[i].type == 0)
		{
			if (particles[i].age > particles[i].life || particles[i].rbt.getTranslation()[1] > 35 || particles[i].rbt.getTranslation()[1] < -25 || particles[i].rbt.getTranslation()[0] > 40 || particles[i].rbt.getTranslation()[0] < -40)
			{
				int prob = rand() % 100;
				if (prob < 10)
				{
					Smoke_conversion(&particles[i]);
				}
				else
				{
					initParticleAttributes(&particles[i]);
				}
			}
		}
		else
		{
			if (particles[i].age > particles[i].life || particles[i].rbt.getTranslation()[1] > 45 || particles[i].rbt.getTranslation()[1] < -35 || particles[i].rbt.getTranslation()[0] > 80 || particles[i].rbt.getTranslation()[0] < -80)
			{
				particles[i].type = 0;
				initParticleAttributes(&particles[i]);
			}
		}

	}

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

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	updateParticles();
	for (int i = 0; i < MaxParticles; i++) {
		Matrix4 MVM = rigTFormToMatrix(invEyeRbt * particles[i].rbt) * Matrix4::makeScale(Cvec3(0.02, 0.02, 0.02));
		sendModelViewNormalMatrix(curSS, MVM, normalMatrix(MVM));
		safe_glUniform3f(curSS.h_uColor, particles[i].color[0], particles[i].color[1], particles[i].color[2]); // set color to grayish
		safe_glUniform1f(curSS.h_uTransparency, 1 - particles[i].age / particles[i].life);
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
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(0.);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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
