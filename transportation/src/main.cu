#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <math.h>
#include <mutex>
#include <vector>   

#include "flamegpu/flamegpu.h"

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <GL/freeglut.h>
#include <GL/glut.h>

#include "clustering\fastcluster.h"

// Grid Size ����������� ����������� ������������ � ��������
#define GRID_WIDTH 800
#define GRID_HEIGHT 800

#define TIME_STOP 3600 // ���������� �������� ������� (� ��������, ����� T=3600 = 1 ���)
#define RUN_COUNT 100 // ���������� �������� ��� ��������
#define DEVISOR 1000000. // ���������, �������� �������� ���������� ��� ������������ ����������
# define M_PI           3.14159265358979323846

// Visualisation mode (1=standalone run, 0 = essemble run, 3 - variation experiments)
#define VIS_MODE 1


#define INTERACTION_RADIUS 150 // ������ ��� ������ ���������
#define PERSONAL_RADIUS 30 // ������ ������� ������������ ������, �������������� �������������� ���������� � ���������� ��������


std::atomic<unsigned int> DRN = { 0 }; // ��� ���
std::atomic <unsigned int> L = { 0 }; // ����� �����
std::atomic <unsigned int> w = { 0 }; // ������ �����
std::atomic <unsigned int> N_nodes = { 0 }; // ���������� ������� ���
std::atomic <unsigned int> R1 = { 0 }; // ���������� ������ ���� ��������� ��������
std::atomic <unsigned int> R2 = { 0 }; 
std::atomic <unsigned int> R3 = { 0 };
std::atomic <unsigned int> R4 = { 0 };
std::atomic <unsigned int> R5 = { 0 };

unsigned int indent_x; // ������ �� x
unsigned int indent_y; // ������ �� y 


std::mutex m;

int window_width = 1050;
int window_height = 1050;
int window_id = 0;

const long MAX_AGENTS = 100000; //������������ ���������� �������

float x_a[MAX_AGENTS], y_a[MAX_AGENTS]; //���������� �������
float rot_a[MAX_AGENTS]; // ���� �������� ��
int agent_class[MAX_AGENTS]; // ����� ������ (��� � ���)
float r_a[MAX_AGENTS]; // ������ ������� ������������ �������
unsigned int agent_state[MAX_AGENTS]; // 

int a_size = 0; // ���������� �������

int jam_count = 0; // ���������� �������� �������
int jam_size = 0; // ��������� �������� �������
int density_jam[1000]; // ��������� �������� �������
float x_jam[1000], y_jam[1000]; //���������� �������� �������


std::ofstream out("results.txt", std::ios::app);
std::ofstream out2("log.txt", std::ios::app); // ��� ��� �������� �������

void display(void);

extern void initVisualisation();
extern void runVisualisation();

__shared__ unsigned int agent_nextID;

__host__ __device__  unsigned int getNextID() {
    agent_nextID++;
    return agent_nextID;
}

//������ ������� ������������ ������-��
inline __device__ double new_radius(double density, double gamma)
{
    return  gamma* (PERSONAL_RADIUS / pow(density, 0.2));
}
//��������� ����� ��������
inline __device__ double distance_agents(double x1, double x2, double y1, double y2)
{
	return  pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5);
}

class Points_ugvs {
public:
    double x;
    double y;
    Points_ugvs(double xx = 0.0, double yy = 0.0) { x = xx; y = yy; }
    Points_ugvs(const Points_ugvs& p) { x = p.x; y = p.y; }
    double norm() { return(sqrt(x * x + y * y)); }
};

// ���������� ��������� ����� ���
__host__ double distance(const  Points_ugvs& s1, const  Points_ugvs& s2) {
    return sqrt(pow(s2.x - s1.x, 2) + pow(s2.y - s1.y, 2));
}

//��������� ��� �������� ������ � ��������� ���
struct CLUSTERS_UGVs {
public:
    double  x_c;
    double  y_c;
    int  d_c;
    CLUSTERS_UGVs(double xx = 0.0, double yy = 0.0, int dc = 0) { x_c = xx; y_c = yy; d_c = dc;  }
};

bool my_clusters_comparison(const CLUSTERS_UGVs& a, const CLUSTERS_UGVs& b)
{
    return a.d_c > b.d_c;
}



GLuint  textura_id1, textura_id2;

struct textura_struct
{
    int W;
    int H;
    unsigned char* Image;
}get_textura;

int LoadTexture(char* FileName, GLuint &textura_id)
{
    FILE* F;
    /* ��������� ���� */
    if ((F = fopen(FileName, "rb")) == NULL)
        return 0;
    /*������������ � bmp-����� �� ������ �������, � ��������� ������ � ������ */
    fseek(F, 18, SEEK_SET);
    fread(&(get_textura.W), 2, 1, F);
    fseek(F, 2, SEEK_CUR);
    fread(&(get_textura.H), 2, 1, F);

    //printf("%d x %d\n", get_textura.W, get_textura.H);

    /* �������� ������ ��� �����������. ���� ������ �� ����������, ��������� ���� � ������� � ������� */
    if ((get_textura.Image = (unsigned char*)malloc(sizeof(unsigned char) * 3 * get_textura.W * get_textura.H)) == NULL)
    {
        fclose(F);
        return 0;
    }
    /* ��������� ����������� � ������ �� 3 ����, �� ���� RGB ��� ������� ������� */
    fseek(F, 30, SEEK_CUR);
    fread(get_textura.Image, 3, get_textura.W * get_textura.H, F);

    glGenTextures(1, &textura_id);
    glBindTexture(GL_TEXTURE_2D, textura_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 3, get_textura.W, get_textura.H, GL_RGB, GL_UNSIGNED_BYTE, get_textura.Image);
    free(get_textura.Image);
    fclose(F);

    return 1;
}

int initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
        "GL_ARB_pixel_buffer_object"
    )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return FALSE;
    }
    // default initialization
    glClearColor(1.0, 1.0, 1.0, 1.0);
  
}

void reshape(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void windowResize(int width, int height) {
    window_width = width;
    window_height = height;
}

extern void initVisualisation()
{
    // Create GL context
    int   argc = 1;
    char glutString[] = "GLUT application";
    char* argv[] = { glutString, NULL };
    //char *argv[] = {"GLUT application", NULL};	

    glutInit(&argc, argv);


    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(window_width, window_height);
    window_id = glutCreateWindow("FLAME GPU Visualiser");
    glutReshapeFunc(windowResize);

    // initialize GL
    if (FALSE == initGL()) {
        return;
    }

    if (LoadTexture((char*)"mgv.bmp", textura_id1) != 1) { printf("�� ������� ��������� �����������\n"); }
    if (LoadTexture((char*)"ugv.bmp", textura_id2) != 1) { printf("�� ������� ��������� �����������\n"); }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    //����� ����� ��������������������
}

extern void runVisualisation()
{
    // start rendering mainloop
    glutMainLoop();


}

//������� ��������� ����������
void drawCircle(float x, float y, float r, int amountSegments)
{
    glColor3d(0, 0, 0);
    glLineWidth(2.0f);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBegin(GL_LINE_LOOP);
    
    for (int i = 0; i < amountSegments; i++)
    {
        float angle = 2.0 * 3.1415926 * float(i) / float(amountSegments);
        float dx = r * cosf(angle);
        float dy = r * sinf(angle);
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
}

void drawCircle2(float x, float y, float r, int amountSegments)
{
    glColor3d(0, 0, 0);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    glPushAttrib(GL_ENABLE_BIT);
    glLineWidth(1);
    glLineStipple(1, 0xAAAA);
    glEnable(GL_LINE_STIPPLE);
    glBegin(GL_LINE_LOOP);

    for (int i = 0; i < amountSegments; i++)
    {
        float angle = 2.0 * 3.1415926 * float(i) / float(amountSegments);
        float dx = r * cosf(angle);
        float dy = r * sinf(angle);
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
    glDisable(GL_LINE_STIPPLE);
}

void drawCircle3(float x, float y, float r, int pointCount, int level)
{

    /*
    if (level == 1)
    glColor4f(0.0, 1.0, 0.0, 0.5); //�������
    if (level == 2)
    glColor4f(1.0, 1.0, 0.0, 0.5); //������
    if (level == 3)
    glColor4f(1.0, 0.0, 0.0, 0.5); //�������
    */

    if (level == 1)
        glColor4f(0.2, 0.2, 0.2, 0.5); //�����
    if (level == 2)
        glColor4f(0.2, 0.2, 0.2, 0.75); //�����
    if (level == 3)
        glColor4f(0.2, 0.2, 0.2, 0.9); //�����

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPushAttrib(GL_ENABLE_BIT);
    glLineWidth(4);
    glLineStipple(1, 0xAAAA);
    glEnable(GL_LINE_STIPPLE);

    glBegin(GL_TRIANGLE_FAN);

    const float step = float(2 * M_PI) / pointCount;
    for (float angle = 0; angle <= float(2 * M_PI); angle += step)
    {
        float a = (fabsf(angle - float(2 * M_PI)) < 0.00001f) ? 0.f : angle;
        const float dx = r * cosf(a);
        const float dy = r * sinf(a);
        glVertex2f(dx + x, dy + y);
    }
    glEnd();
    glDisable(GL_LINE_STIPPLE);
}

void display()
{
    glutSetWindow(window_id);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //������������ �������� �������� ����

    

    if (DRN == 1) // ���������-��������� � ����� ����� ��������� ��������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        double R1 = 300;
        double R2 = R1 + w / 2;
        double R3 = R2 + w / 2;
        */

        // ������ ������ (��������������)************************************************************
        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        //������� ������ ������
        glVertex2f(indent_x, indent_y + L/2 + w);
        glVertex2f(indent_x + L, indent_y + L/2 + w);

        //�������� ��������������
        glVertex2f(indent_x, indent_y + L/2 );
        glVertex2f(indent_x + L, indent_y + L/2);

        //������� ������ ������
        glVertex2f(indent_x, indent_y + L/2 - w);
        glVertex2f(indent_x + L, indent_y + L/2 - w);

        glEnd();

        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(indent_x, indent_y + L / 2 + w / 2);
        glVertex2f(indent_x + L, indent_y + L / 2 + w / 2);

        glVertex2f(indent_x, indent_y + L / 2 - w / 2);
        glVertex2f(indent_x + L, indent_y + L / 2 - w / 2);

        glEnd();
        glDisable(GL_LINE_STIPPLE);

        // ������ ������ (������������)**********************************************************************************

        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        //������� ������ ������
        glVertex2f(indent_x + L / 2 - w, indent_y);
        glVertex2f(indent_x + L / 2 - w, indent_y + L);

        //�������� ��������������
        glVertex2f(indent_x + L / 2, indent_y);
        glVertex2f(indent_x + L / 2, indent_y + L);

        //������� ������ ������
        glVertex2f(indent_x + L / 2 + w, indent_y);
        glVertex2f(indent_x + L / 2 + w, indent_y + L);

        glEnd();

        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(indent_x + L/2 - w / 2, indent_y);
        glVertex2f(indent_x + L / 2 - w / 2, indent_y + L);

        glVertex2f(indent_x + L / 2 + w / 2, indent_y);
        glVertex2f(indent_x + L / 2 + w / 2, indent_y + L);

        glEnd();
        glDisable(GL_LINE_STIPPLE);


        //���� ��������� ��������*********************************************
        //���������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R1, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R2, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R3, 5000);

    }

    if (DRN == 2) // ���������-��������� � ����� ������ ��������� ��������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        double R1 = 250;
        double R2 = R1 + w / 2;
        double R3 = R2 + w / 2;
        double R4 = R3 + w / 2;
        double R5 = R4 + w / 2;
        */
        // ������ ������ (��������������)************************************************************
        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        //������� ������ ������
        glVertex2f(indent_x, indent_y + L / 2 + w);
        glVertex2f(indent_x + L, indent_y + L / 2 + w);

        //�������� ��������������
        glVertex2f(indent_x, indent_y + L / 2);
        glVertex2f(indent_x + L, indent_y + L / 2);

        //������� ������ ������
        glVertex2f(indent_x, indent_y + L / 2 - w);
        glVertex2f(indent_x + L, indent_y + L / 2 - w);

        glEnd();

        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(indent_x, indent_y + L / 2 + w / 2);
        glVertex2f(indent_x + L, indent_y + L / 2 + w / 2);

        glVertex2f(indent_x, indent_y + L / 2 - w / 2);
        glVertex2f(indent_x + L, indent_y + L / 2 - w / 2);

        glEnd();
        glDisable(GL_LINE_STIPPLE);

        // ������ ������ (������������)**********************************************************************************

        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        //������� ������ ������
        glVertex2f(indent_x + L / 2 - w, indent_y);
        glVertex2f(indent_x + L / 2 - w, indent_y + L);

        //�������� ��������������
        glVertex2f(indent_x + L / 2, indent_y);
        glVertex2f(indent_x + L / 2, indent_y + L);

        //������� ������ ������
        glVertex2f(indent_x + L / 2 + w, indent_y);
        glVertex2f(indent_x + L / 2 + w, indent_y + L);

        glEnd();

        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(indent_x + L / 2 - w / 2, indent_y);
        glVertex2f(indent_x + L / 2 - w / 2, indent_y + L);

        glVertex2f(indent_x + L / 2 + w / 2, indent_y);
        glVertex2f(indent_x + L / 2 + w / 2, indent_y + L);

        glEnd();
        glDisable(GL_LINE_STIPPLE);


        //���� ��������� ��������*********************************************
        //���������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R1, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R2, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R3, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R4, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R5, 5000);

    }
    
    if (DRN == 3) // �������������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        int N_nodes = 2; // ���������� ������� ���
        */

        // �������������� ������ ************************************************************
        double y_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial + 2 * w;

        }

        y_initial = 0;
        for (int i = 0; i < N_nodes; i++)

        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial - 2 * w;

        }

        // ������������ ������ **********************************************************************************
        double x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial + 2 * w;
        }
        
        x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial - 2 * w;
        }


    }

    if (DRN == 4) // ������������-������������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        int N_nodes = 2; // ���������� ������� ���
        */


        // �������������� ������ ************************************************************
        double y_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial +  L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial + 2 * w;

        }


        y_initial = 0;
        for (int i = 0; i < N_nodes; i++)

        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial - 2 * w;

        }
       
        // ������������ ������ **********************************************************************************
       
        double x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial + 2 * w;
        }

        x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial - 2 * w;
        }
        

        // ���������******************************************************************

        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        // �����-�������
        //������� ������ ������
        glVertex2f(indent_x + (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L + (double)w / sin(45 * M_PI / 180), indent_y);

        //�������� �������������� �����
        glVertex2f(indent_x, indent_y + L);
        glVertex2f(indent_x + L, indent_y);

        //������� ������ ������
        glVertex2f(indent_x - (double) w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L - (double) w / sin(45 * M_PI / 180), indent_y);

        // ������-������
        
        //������� ������ ������
        glVertex2f(indent_x + L - (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x - (double)w / sin(45 * M_PI / 180), indent_y);

        //�������� �������������� �����
        glVertex2f(indent_x + L, indent_y + L);
        glVertex2f(indent_x, indent_y);

        //������� ������ ������
        glVertex2f(indent_x + L + (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + (double)w / sin(45 * M_PI / 180), indent_y);

        glEnd();


        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);

        // �����-�������
        glVertex2f(indent_x + (double) (w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L + (double)(w / 2) / sin(45 * M_PI / 180), indent_y);

        glVertex2f(indent_x - (double) (w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L - (double) (w / 2) / sin(45 * M_PI / 180), indent_y);

        // ������-������
        glVertex2f(indent_x + L - (double) (w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x - (double) (w / 2) / sin(45 * M_PI / 180), indent_y);

        glVertex2f(indent_x + L + (double) (w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + (double) (w / 2) / sin(45 * M_PI / 180), indent_y);

        glEnd();
    }

    if (DRN == 5) // �����������-���������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        double R1 = 200;
        double R2 = R1 + w / 2;
        double R3 = R2 + w / 2;
        double R4 = R3 + w / 2;
        double R5 = R4 + w / 2;
        int N_nodes = 2; // ���������� ������� ���
        */

        // �������������� ������ ************************************************************
        double y_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, y_initial + L / 2 + w);
            glVertex2f(indent_x + L, y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, y_initial + L / 2);
            glVertex2f(indent_x + L, y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, y_initial + L / 2 - w);
            glVertex2f(indent_x + L, y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial + 2 * w;

        }

        y_initial = 0;
        for (int i = 0; i < N_nodes; i++)

        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, y_initial + L / 2 + w);
            glVertex2f(indent_x + L, y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, y_initial + L / 2);
            glVertex2f(indent_x + L, y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, y_initial + L / 2 - w);
            glVertex2f(indent_x + L, y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial - 2 * w;

        }

        // ������������ ������ **********************************************************************************
        double x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial + 2 * w;
        }

        x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial - 2 * w;
        }

        //���� ��������� ��������*********************************************
        //���������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R1, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R2, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R3, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R4, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R5, 5000);


    }
     
    if (DRN == 6) // ���������������: ������������-�����������-���������
    {
        /*
        double L = 1000; // ����� �����
        double w = 100; // ������ �����
        int N_nodes = 2; // ���������� ������� ���
        double R1 = 200;
        double R2 = R1 + w / 2;
        double R3 = R2 + w / 2;
        double R4 = R3 + w / 2;
        double R5 = R4 + w / 2;
        */

        // �������������� ������ ************************************************************
        double y_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial + 2 * w;

        }


        y_initial = 0;
        for (int i = 0; i < N_nodes; i++)

        {

            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w);

            //�������� ��������������
            glVertex2f(indent_x, indent_y + y_initial + L / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2);

            //������� ������ ������
            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x, indent_y + y_initial + L / 2 + w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 + w / 2);

            glVertex2f(indent_x, indent_y + y_initial + L / 2 - w / 2);
            glVertex2f(indent_x + L, indent_y + y_initial + L / 2 - w / 2);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            y_initial = y_initial - 2 * w;

        }

        // ������������ ������ **********************************************************************************

        double x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial + 2 * w;
        }

        x_initial = 0;
        for (int i = 0; i < N_nodes; i++)
        {
            glLineWidth(2);       // ������ �����
            glBegin(GL_LINES);
            glColor3d(0.0, 0.0, 0.0);     // ������ ����

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w, indent_y + L);

            //�������� ��������������
            glVertex2f(indent_x + x_initial + L / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2, indent_y + L);

            //������� ������ ������
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w, indent_y + L);

            glEnd();

            //���������� ������ ������
            glPushAttrib(GL_ENABLE_BIT);
            glLineWidth(1);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 - w / 2, indent_y + L);

            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y);
            glVertex2f(indent_x + x_initial + L / 2 + w / 2, indent_y + L);

            glEnd();
            glDisable(GL_LINE_STIPPLE);

            x_initial = x_initial - 2 * w;
        }


        // ���������******************************************************************

        glLineWidth(2);       // ������ �����
        glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);     // ������ ����

        // �����-�������
        //������� ������ ������
        glVertex2f(indent_x + (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L + (double)w / sin(45 * M_PI / 180), indent_y);

        //�������� �������������� �����
        glVertex2f(indent_x, indent_y + L);
        glVertex2f(indent_x + L, indent_y);

        //������� ������ ������
        glVertex2f(indent_x - (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L - (double)w / sin(45 * M_PI / 180), indent_y);

        // ������-������

        //������� ������ ������
        glVertex2f(indent_x + L - (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x - (double)w / sin(45 * M_PI / 180), indent_y);

        //�������� �������������� �����
        glVertex2f(indent_x + L, indent_y + L);
        glVertex2f(indent_x, indent_y);

        //������� ������ ������
        glVertex2f(indent_x + L + (double)w / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + (double)w / sin(45 * M_PI / 180), indent_y);

        glEnd();


        //���������� ������ ������
        glPushAttrib(GL_ENABLE_BIT);
        glLineWidth(1);
        glLineStipple(1, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);

        // �����-�������
        glVertex2f(indent_x + (double)(w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L + (double)(w / 2) / sin(45 * M_PI / 180), indent_y);

        glVertex2f(indent_x - (double)(w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + L - (double)(w / 2) / sin(45 * M_PI / 180), indent_y);

        // ������-������
        glVertex2f(indent_x + L - (double)(w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x - (double)(w / 2) / sin(45 * M_PI / 180), indent_y);

        glVertex2f(indent_x + L + (double)(w / 2) / sin(45 * M_PI / 180), indent_y + L);
        glVertex2f(indent_x + (double)(w / 2) / sin(45 * M_PI / 180), indent_y);

        glEnd();


        //���� ��������� ��������*********************************************
        //���������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R1, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R2, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R3, 5000);
        //���������� �� ������
        drawCircle2(indent_x + L / 2, indent_y + L / 2, R4, 5000);
        //������� ������
        drawCircle(indent_x + L / 2, indent_y + L / 2, R5, 5000);


    }
    
    for (int i = 0; i < a_size; i++)
    {
        glPointSize(4);

       // glBegin(GL_POINTS);

               
        if (agent_class[i] == 1 && agent_state[i] != 1) // ���
        {
            /* ����� ����������� � ���� */
            glPushMatrix();
          
            glEnable(GL_TEXTURE_2D);
            glColor3f(1, 1, 1);

            glBindTexture(GL_TEXTURE_2D, textura_id1); // ���

            glTranslatef(x_a[i], y_a[i], 0);
            glRotatef(rot_a[i] * 180 / M_PI, 0.0f, 0.0f, 1.0f); //������� ��
            glTranslatef(-x_a[i], -y_a[i], 0);

            glBegin(GL_QUADS);
            glTexCoord2d(0, 0); glVertex2d( (x_a[i] - 15), (y_a[i] - 5) );
            glTexCoord2d(0, 1); glVertex2d( (x_a[i] - 15), (y_a[i] + 5) );
            glTexCoord2d(1, 1); glVertex2d( (x_a[i] + 15), (y_a[i] + 5) );
            glTexCoord2d(1, 0); glVertex2d( (x_a[i] + 15), (y_a[i] - 5) );
            glEnd();
            
            glDisable(GL_TEXTURE_2D);
            glPopMatrix();
        }

        if (agent_class[i] == 2 && agent_state[i] != 1) // ���
        {
            /* ����� ����������� � ���� */
            glPushMatrix();
            
            glEnable(GL_TEXTURE_2D);
            glColor3f(1, 1, 1);

            glBindTexture(GL_TEXTURE_2D, textura_id2); // ���
            
            glTranslatef(x_a[i], y_a[i], 0);
            glRotatef(rot_a[i] * 180/ M_PI, 0.0f, 0.0f, 1.0f); //������� ��
            glTranslatef(-x_a[i], -y_a[i], 0);

            glBegin(GL_QUADS);
            glTexCoord2d(0, 0); glVertex2d(x_a[i] - 15, y_a[i] - 5);
            glTexCoord2d(0, 1); glVertex2d(x_a[i] - 15, y_a[i] + 5);
            glTexCoord2d(1, 1); glVertex2d(x_a[i] + 15, y_a[i] + 5);
            glTexCoord2d(1, 0); glVertex2d(x_a[i] + 15, y_a[i] - 5);
            glEnd();

            glDisable(GL_TEXTURE_2D);
            glPopMatrix();
           
        }

        if (agent_state[i] == 1) // �� � ��������� ���������
        {
            /* ����� ����������� � ���� */
            glPushMatrix();

            glEnable(GL_TEXTURE_2D);
            glColor3f(0, 0, 0);

            glBindTexture(GL_TEXTURE_2D, textura_id2); // ���

            glTranslatef(x_a[i], y_a[i], 0);
            glRotatef(rot_a[i] * 180 / M_PI, 0.0f, 0.0f, 1.0f); //������� ��
            glTranslatef(-x_a[i], -y_a[i], 0);

            glBegin(GL_QUADS);
            glTexCoord2d(0, 0); glVertex2d(x_a[i] - 15, y_a[i] - 5);
            glTexCoord2d(0, 1); glVertex2d(x_a[i] - 15, y_a[i] + 5);
            glTexCoord2d(1, 1); glVertex2d(x_a[i] + 15, y_a[i] + 5);
            glTexCoord2d(1, 0); glVertex2d(x_a[i] + 15, y_a[i] - 5);
            glEnd();

            glDisable(GL_TEXTURE_2D);
            glPopMatrix();
        }

        //glVertex3d(x_a[i], y_a[i], 0); // ������-�����
        //glEnd();

        drawCircle2(x_a[i], y_a[i], r_a[i], 100); // ������ ������� ������������ ��

    }

       
    //������������ "������"
    for (int i = 0; i < jam_count; i++)
    {
        //��������� ���������� ���������
        glPointSize(10);
        glBegin(GL_POINTS);
        glColor4f(1.0f, 0.5f, 0.0f, 0.0f);//orange/brown
        glVertex3d(x_jam[i], y_jam[i], 0); // ������ ���������
        glEnd();

        int lvl = 1;
        if (density_jam[i] < 6)
            lvl = 1;
        if (density_jam[i] >= 6 && density_jam[i] < 10)
            lvl = 2;
        if (density_jam[i] >= 10)
            lvl = 3;

        drawCircle3(x_jam[i], y_jam[i], PERSONAL_RADIUS *3, 100, lvl); 

    }

  
    
    
    //redraw

glutSwapBuffers();
glutPostRedisplay();

}


void timer(int = 0)
{
    display();
    glutTimerFunc(1, timer, 0);
}

std::atomic_int par1; //��������� ������������ ������ ���
std::atomic_int par2;
std::atomic_int objective; //������� ������� ���


FLAMEGPU_INIT_FUNCTION(init_function) {
    std::lock_guard<std::mutex> lock(m);

    //�������� ������������ ���*************************************************************************
    DRN = FLAMEGPU->environment.getProperty<unsigned int>("DRN");
    L = FLAMEGPU->environment.getProperty<unsigned int>("L");
    w = FLAMEGPU->environment.getProperty<unsigned int>("w");
    N_nodes = FLAMEGPU->environment.getProperty<unsigned int>("N_nodes");
    R1 = FLAMEGPU->environment.getProperty<unsigned int>("R1");
    R2 = R1 + w / 2;
    R3 = R2 + w / 2;
    R4 = R3 + w / 2;
    R5 = R4 + w / 2;


    indent_x= FLAMEGPU->environment.getProperty<unsigned int>("indent_x"); // ������� �� x � y ��� ���
    indent_y = FLAMEGPU->environment.getProperty<unsigned int>("indent_y");
}

FLAMEGPU_STEP_FUNCTION(BasicOutput) {
    std::lock_guard<std::mutex> lock(m);

    flamegpu::HostAgentAPI agent = FLAMEGPU->agent("agent-vehicles");

    DRN = FLAMEGPU->environment.getProperty<unsigned int>("DRN");
    
    //��������� �������-��� � ���������� �� � ���
    int direction = 0;
    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;
    double x3 = 0;
    double y3 = 0;
    double v = 0;

    int x_min = indent_x;
    int x_max = x_min;
    int y_min = indent_y + L / 2;
    int y_max = indent_y + L / 2 + w;

    int x_min2 = indent_x;
    int x_max2 = x_min2;
    int y_min2 = indent_y + L / 2 - 2 * w;
    int y_max2 = indent_y + L / 2 - w;

    int x_min3 = indent_x;
    int x_max3 = x_min3;
    int y_min3 = indent_y + L / 2 + 2 * w;
    int y_max3 = indent_y + L / 2 + 3 * w;


    int x_min4 = indent_x;
    int x_max4 = x_min4 + (double) w * cos(45 * M_PI / 180);
    int y_min4 = indent_y;
    int y_max4 = indent_y + (double)w * sin(45 * M_PI / 180);

    int unsigned intensity_of_UGVs = FLAMEGPU->environment.getProperty<unsigned int>("intensity_of_UGVs");

    if (VIS_MODE == 3)
        intensity_of_UGVs = par1;

    if (FLAMEGPU->getStepCounter() % FLAMEGPU->environment.getProperty<unsigned int>("frequency") == 0)
    {
        int n = 0;
        while (n < intensity_of_UGVs)
        {
            
            if (DRN == 1 || DRN == 2 || DRN == 3 ||  DRN == 5)
            direction = FLAMEGPU->random.uniform<int>(1, 4);

            if (DRN == 4 || DRN == 6)
            direction = FLAMEGPU->random.uniform<int>(1, 8);


            switch (direction) {
            case 1: // �����-�������
                x_min = indent_x;
                x_max = x_min;
                y_min = indent_y + L / 2 - w;
                y_max = indent_y + L / 2;

                if(DRN==3 || DRN == 4 || DRN == 5 || DRN == 6)
                    { 
                    x_min2 = indent_x;
                    x_max2 = x_min2;
                    y_min2 = indent_y + L / 2 - 3 * w;
                    y_max2 = indent_y + L / 2 - 2 * w;

                    x_min3 = indent_x;
                    x_max3 = x_min3;
                    y_min3 = indent_y + L / 2 + w;
                    y_max3 = indent_y + L / 2 + 2 * w;
                    }

                break;
            case 2: // ������-������
                x_min = indent_x + L;
                x_max = x_min;

                y_min = indent_y + L / 2;
                y_max = indent_y + L / 2 + w;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L;
                    x_max2 = x_min2;
                    y_min2 = indent_y + L / 2 - 2 * w;
                    y_max2 = indent_y + L / 2 - w;

                    x_min3 = indent_x + L;
                    x_max3 = x_min3;
                    y_min3 = indent_y + L / 2 + 2 * w;
                    y_max3 = indent_y + L / 2 + 3 * w;
                }

                break;
            case 3:  // �����-�����
                x_min = indent_x + L / 2;
                x_max = indent_x + L / 2 + w;
                y_min = indent_y;
                y_max = y_min;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L / 2 - 2 * w;
                    x_max2 = indent_x + L / 2 - w;
                    y_min2 = indent_y;
                    y_max2 = y_min2;

                    x_min3 = indent_x + L / 2 + 2 * w;
                    x_max3 = indent_x + L / 2 + 3 * w;
                    y_min3 = indent_y;
                    y_max3 = y_min3;
                }

                break;
            case 4: // ������-����
                x_min = indent_x + L / 2 - w;
                x_max = indent_x + L / 2;
                y_min = indent_y + L;
                y_max = y_min;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L / 2 - 3 * w;
                    x_max2 = indent_x + L / 2 - 2 * w;
                    y_min2 = indent_y + L;
                    y_max2 = y_min2;

                    x_min3 = indent_x + L / 2 + w;
                    x_max3 = indent_x + L / 2 + 2 * w;
                    y_min3 = indent_y + L;
                    y_max3 = y_min3;
                }

                break;
            case 5: // ���������  �-�
                   x_min = indent_x;
                   x_max = x_min + (double)w * cos (45 * M_PI / 180);
                   y_min = indent_y + w;
                   y_max = y_min + (double)w * sin(45 * M_PI / 180);
               
                   break;
            case 6: // ���������  �-�
                   x_min = indent_x + L / 2 + 4 * w;
                   x_max = x_min + (double)w * cos(45 * M_PI / 180);
                   y_min = indent_y + L / 2 + 4 * w;
                   y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;

            case 7: // ���������  �-�
                x_min = indent_x;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + L / 2 + 4 * w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;

            case 8: // ���������  �-�
                x_min = indent_x + L / 2 + 4 * w;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;
            }
            
            
            x = FLAMEGPU->random.uniform(x_min, x_max);
            y = FLAMEGPU->random.uniform(y_min, y_max);

            
            // �� ��� ������ � ������ ����������� (�.�., �����-�������, ������-������, ������-���� � �.�.)
            if ( (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6) && (direction==1 || direction == 2 || direction == 3 || direction == 4) )
            {
                int share = FLAMEGPU->random.uniform<int>(1, 3);
                x2 = FLAMEGPU->random.uniform(x_min2, x_max2);
                y2 = FLAMEGPU->random.uniform(y_min2, y_max2);
                x3 = FLAMEGPU->random.uniform(x_min3, x_max3);
                y3 = FLAMEGPU->random.uniform(y_min3, y_max3);

                if (share == 1)
                {
                    x = x;
                    y = y;
                }
                if (share == 2)
                {
                    x = x2;
                    y = y2;
                }
                if (share == 3)
                {
                    x = x3;
                    y = y3;
                }
            }

            float vv = FLAMEGPU->environment.getProperty<float>("velocity_of_UGVs");

            if(VIS_MODE==3)
                vv = par2;

            v = FLAMEGPU->random.logNormal<float>(log(vv), 0.1); // ��������� �������� ���
           
            flamegpu::HostNewAgentAPI instance = agent.newAgent();
            instance.setVariable<int>("id", getNextID());
            instance.setVariable<float>("x", x);
            instance.setVariable<float>("y", y);
            instance.setVariable<float>("Ra", PERSONAL_RADIUS);
            instance.setVariable<int>("agent_class", 1);
            instance.setVariable<int>("agent_state", 2);
            instance.setVariable<int>("agent_type", direction);
            instance.setVariable<float>("velocity", v);
            instance.setVariable<float>("rotation", 0);
            instance.setVariable<float>("neighbour_distance", 10000);
            instance.setVariable<float>("other_neighbour_distance", 10000);
            n++;
        }

        n = 0;
        while (n < FLAMEGPU->environment.getProperty<unsigned int>("intensity_of_MGVs"))
        {
            if (DRN == 1 || DRN == 2 || DRN == 3 || DRN == 5)
                direction = FLAMEGPU->random.uniform<int>(1, 4);

            if (DRN == 4 || DRN == 6)
                direction = FLAMEGPU->random.uniform<int>(1, 8);

            switch (direction) {
            case 1: // �����-�������
                x_min = indent_x;
                x_max = x_min;
                y_min = indent_y + L / 2 - w;
                y_max = indent_y + L / 2;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x;
                    x_max2 = x_min2;
                    y_min2 = indent_y + L / 2 - 3 * w;
                    y_max2 = indent_y + L / 2 - 2 * w;

                    x_min3 = indent_x;
                    x_max3 = x_min3;
                    y_min3 = indent_y + L / 2 + w;
                    y_max3 = indent_y + L / 2 + 2 * w;
                }

                break;
            case 2: // ������-������
                x_min = indent_x + L;
                x_max = x_min;

                y_min = indent_y + L / 2;
                y_max = indent_y + L / 2 + w;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L;
                    x_max2 = x_min2;
                    y_min2 = indent_y + L / 2 - 2 * w;
                    y_max2 = indent_y + L / 2 - w;

                    x_min3 = indent_x + L;
                    x_max3 = x_min3;
                    y_min3 = indent_y + L / 2 + 2 * w;
                    y_max3 = indent_y + L / 2 + 3 * w;
                }

                break;
            case 3:  // �����-�����
                x_min = indent_x + L / 2;
                x_max = indent_x + L / 2 + w;
                y_min = indent_y;
                y_max = y_min;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L / 2 - 2 * w;
                    x_max2 = indent_x + L / 2 - w;
                    y_min2 = indent_y;
                    y_max2 = y_min2;

                    x_min3 = indent_x + L / 2 + 2 * w;
                    x_max3 = indent_x + L / 2 + 3 * w;
                    y_min3 = indent_y;
                    y_max3 = y_min3;
                }

                break;
            case 4: // ������-�����
                x_min = indent_x + L / 2 - w;
                x_max = indent_x + L / 2;
                y_min = indent_y + L;
                y_max = y_min;

                if (DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6)
                {
                    x_min2 = indent_x + L / 2 - 3 * w;
                    x_max2 = indent_x + L / 2 - 2 * w;
                    y_min2 = indent_y + L;
                    y_max2 = y_min2;

                    x_min3 = indent_x + L / 2 + w;
                    x_max3 = indent_x + L / 2 + 2 * w;
                    y_min3 = indent_y + L;
                    y_max3 = y_min3;
                }

                break;

            case 5: // ���������  �-�
                x_min = indent_x;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;
            case 6: // ���������  �-�
                x_min = indent_x + L / 2 + 5 * w;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + L / 2 + 4 * w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;

            case 7: // ���������  �-�
                x_min = indent_x;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + L / 2 + 4 * w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;

            case 8: // ���������  �-�
                x_min = indent_x + L / 2 + 4 * w;
                x_max = x_min + (double)w * cos(45 * M_PI / 180);
                y_min = indent_y + w;
                y_max = y_min + (double)w * sin(45 * M_PI / 180);

                break;

            }

                      
            x = FLAMEGPU->random.uniform(x_min, x_max);
            y = FLAMEGPU->random.uniform(y_min, y_max);

            // �� ��� ������ � ������ ����������� (�.�., �����-�������, ������-������, ������-���� � �.�.)
            if ((DRN == 3 || DRN == 4 || DRN == 5 || DRN == 6) && (direction == 1 || direction == 2 || direction == 3 || direction == 4))
            {
                int share = FLAMEGPU->random.uniform<int>(1, 3);
                x2 = FLAMEGPU->random.uniform(x_min2, x_max2);
                y2 = FLAMEGPU->random.uniform(y_min2, y_max2);
                x3 = FLAMEGPU->random.uniform(x_min3, x_max3);
                y3 = FLAMEGPU->random.uniform(y_min3, y_max3);

                if (share == 1)
                {
                    x = x;
                    y = y;
                }
                if (share == 2)
                {
                    x = x2;
                    y = y2;
                }
                if (share == 3)
                {
                    x = x3;
                    y = y3;
                }
            }



            float vv = FLAMEGPU->environment.getProperty<float>("velocity_of_MGVs");
            v = FLAMEGPU->random.logNormal<float>(log(vv), 0.1); // ��������� �������� ���

            flamegpu::HostNewAgentAPI instance = agent.newAgent();
            instance.setVariable<int>("id", getNextID());
            instance.setVariable<float>("x", x);
            instance.setVariable<float>("y", y);
            instance.setVariable<float>("Ra", PERSONAL_RADIUS);
            instance.setVariable<int>("agent_class", 2);
            instance.setVariable<int>("agent_state", 2);
            instance.setVariable<int>("agent_type", direction);
            instance.setVariable<float>("velocity", v);
            instance.setVariable<float>("rotation", 0);
            instance.setVariable<float>("neighbour_distance", 10000);
            instance.setVariable<float>("other_neighbour_distance", 10000);
            n++;
        }
    }


}

FLAMEGPU_EXIT_CONDITION(exit_condition) {

    if (FLAMEGPU->getStepCounter() >= TIME_STOP - 1)
        return  flamegpu::EXIT;  // ��������� ���������
    else
        return  flamegpu::CONTINUE;  // ����������� ���������
}


FLAMEGPU_HOST_FUNCTION(agents_data_updating) {

    std::lock_guard<std::mutex> lock(m);
    
    std::vector<Points_ugvs> agent_xy;
    agent_xy.clear();

    flamegpu::HostAgentAPI agent = FLAMEGPU->agent("agent-vehicles");
    
    flamegpu::DeviceAgentVector population1 = agent.getPopulationData();
    
    auto traffic = FLAMEGPU->environment.getMacroProperty<uint32_t>("Traffic");

    unsigned int accidents_count = 0;
    unsigned int traffic_count = traffic;

    for (int i = 0; i < agent.count(); i++)
    {
        flamegpu::AgentVector::Agent instance = population1[i];

        x_a[i] = instance.getVariable<float>("x");
        y_a[i] = instance.getVariable<float>("y");
        r_a[i] = instance.getVariable<float>("Ra");
        rot_a[i] = instance.getVariable<float>("rotation");
        agent_class[i] = instance.getVariable<int>("agent_class");
        agent_state[i] = instance.getVariable<int>("agent_state");

        //if (agent_class[i] == 1 && agent_state[i] != 1)
            agent_xy.push_back(Points_ugvs(x_a[i], y_a[i]));


        if (agent_state[i] == 1)
            accidents_count++;
    }


    objective = accidents_count;

    printf("%u, %u \n", traffic_count, accidents_count);

    a_size = agent.count();
   
    
     //���������� ��������� ������������� ������������� ********************************************************************** 
    
     // clustering call
    int npoints = agent_xy.size();
    if (npoints > 2)
    {

        double* distmat = new double[(npoints * (npoints - 1)) / 2];
        int k, i, j;
        for (i = k = 0; i < npoints; i++) {
            for (j = i + 1; j < npoints; j++) {
                // compute distance between agents 
                distmat[k] = distance(agent_xy[i], agent_xy[j]);
                k++;
            }
        }

        int* merge = new int[2 * (npoints - 1)];
        double* height = new double[npoints - 1];
        hclust_fast(npoints, distmat, HCLUST_METHOD_MEDIAN, merge, height);

        int* labels = new int[npoints];
        //cutree_k(npoints, merge, 2, labels); // 2 - ��������
        cutree_cdist(npoints, merge, height, 2 * PERSONAL_RADIUS, labels); // ����������� ���������� ����� ���������� - ��������� ������ ������� ������������

        std::vector <int> cluster_id;
        cluster_id.clear();
    
        int f = 0;
        for (int i = 0; i < npoints; i++) {
            f = 0;
            for (int j = 0; j < cluster_id.size(); j++) {
                if (labels[i] == cluster_id[j])
                {
                    f = 1; // ������� ������
                    break;
                }
            }
            if (f == 0)
                cluster_id.push_back(labels[i]);
        }

        //���������� ������� ���������, � ������� ����� ���������� ��� � ���������� ���������� � ������-��������
        unsigned int counter = 0;
        std::vector <CLUSTERS_UGVs> ugvs;

        ugvs.clear();
        jam_size = 0;
        jam_count = 0;
        for (int i = 0; i < cluster_id.size(); i++)
        {
            int r = 0;
            double x_cluster = 0;
            double y_cluster = 0;
            int d_cluster = 0;


            for (int j = 0; j < npoints; j++)
            {
                if (cluster_id[i] == labels[j])
                {
                    //���������� ������� ���������
                    x_cluster = x_cluster + agent_xy[j].x;
                    y_cluster = y_cluster + agent_xy[j].y;
                    d_cluster++; // ���������� ������� � ��������
                    r++;
                }

            }

            ugvs.push_back(CLUSTERS_UGVs((double)x_cluster / r, (double)y_cluster / r, d_cluster));

            counter++;
        }

        // ���������� ��������� �� ���������� ��� � ��������
        std::sort(ugvs.begin(), ugvs.end(), my_clusters_comparison);
        
        for (int i = 0; i < ugvs.size(); i++)
        {
            if (ugvs[i].d_c > 3) // ���� �������� ����� �� ����� �� � ��������, �� ��� ������
            {
                x_jam[jam_count] = ugvs[i].x_c;
                y_jam[jam_count] = ugvs[i].y_c;
                density_jam[jam_count] = ugvs[i].d_c; // ��������� �������� �������
                jam_size+= ugvs[i].d_c; // ����� ���������� �������� �������
                jam_count++;
            }
        }

        float avg_jams_density = 0;
        if (jam_count > 0)
        {
            avg_jams_density = (float)jam_size / jam_count;
            printf("TRAFFIC JAMS: %f\n", avg_jams_density);
        }

        if (out.is_open() && VIS_MODE == 1) // �������� ����������� ��� ���������� �������
        {
            out << avg_jams_density << std::endl;
        }
        if (FLAMEGPU->getStepCounter() == TIME_STOP - 1 && VIS_MODE == 1)
            out.close();

        // clean up
        delete[] distmat;
        delete[] merge;
        delete[] height;
        delete[] labels;

    

    }
  
    population1.syncChanges();
   //  population1.purgeCache();
   //population1.purgeCache();
}


FLAMEGPU_AGENT_FUNCTION(agent_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Behaviour goes here

    int direction = FLAMEGPU->getVariable<int>("agent_type");
    FLAMEGPU->setVariable<float>("rotation", 0);

    int x_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_x");
    int y_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_y");

    int R1 = FLAMEGPU->environment.getProperty<unsigned int>("R1");
    int L = FLAMEGPU->environment.getProperty<unsigned int>("L");
    int w = FLAMEGPU->environment.getProperty<unsigned int>("w");
    int R2 = R1 + w / 2;
    int R3 = R2 + w / 2;
    int R4 = R3 + w / 2;
    int R5 = R4 + w / 2;
    int sign = -1;
    int c1 = 100;

    float x_center = x_min + L / 2;
    float y_center = y_min + L / 2;

    double distance = distance_agents(FLAMEGPU->getVariable<float>("x"), x_center, FLAMEGPU->getVariable<float>("y"), y_center);

    double alpha = -0.1 * FLAMEGPU->getVariable<float>("velocity") / distance;
    double beta = atan2((FLAMEGPU->getVariable<float>("y") - y_center), (FLAMEGPU->getVariable<float>("x") - x_center));

    double prob = FLAMEGPU->random.uniform<float>();

    //����������� ������������ �������
    int DRN = FLAMEGPU->environment.getProperty<unsigned int>("DRN");
    if (DRN == 1)
        { 
        if (distance > R1 && distance < R3 && prob < 0.1 && FLAMEGPU->getVariable<int>("agent_state") != 1)
            FLAMEGPU->setVariable<int>("agent_state", 3); // �������� ��������
        }

    if (DRN == 2 || DRN == 5 ||  DRN == 6)
        {
        if(  ((distance > R1 && distance < R3 && prob < 0.05 && FLAMEGPU->getVariable<int>("agent_state") != 1) ||
             (distance > R3 && distance < R5 && prob < 0.05 && FLAMEGPU->getVariable<int>("agent_state") != 1) ) &&
			  FLAMEGPU->getVariable<int>("agent_state")==2)
             
            FLAMEGPU->setVariable<int>("agent_state", 3); // �������� ��������
        }


       //���� ����� ��������� � ���������� ���������
       if(FLAMEGPU->getVariable<int>("agent_state")!=1)
       { 
           
           if (FLAMEGPU->random.uniform<float>() > 0.5)
               sign = 1;

           //���������� ��������� ��
           if (FLAMEGPU->getVariable<float>("neighbour_distance") > 2 * FLAMEGPU->getVariable<float>("threshold_distance") &&
               FLAMEGPU->getVariable<float>("neighbour_angle") < 5) // ��� ����������� �������
           {
               float vv = 0.0;
               if (FLAMEGPU->getVariable<int>("agent_class") == 1)
                   vv = FLAMEGPU->environment.getProperty<float>("velocity_of_UGVs");
               if (FLAMEGPU->getVariable<int>("agent_class") == 2)
                   vv = FLAMEGPU->environment.getProperty<float>("velocity_of_MGVs");

               FLAMEGPU->setVariable<float>("velocity", FLAMEGPU->random.logNormal<float>(log(vv), 0.1));

           }

           if (FLAMEGPU->getVariable<float>("neighbour_distance") <= 2 * FLAMEGPU->getVariable<float>("threshold_distance") &&
               FLAMEGPU->getVariable<float>("neighbour_angle") < 5) // ���� ����������� ������� �� ��
           {
               FLAMEGPU->setVariable<float>("velocity", 0.01 * FLAMEGPU->getVariable<float>("velocity"));
               //printf("Deceleration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
           }

           if (DRN == 1) // ������ ��� ����� ���
           { 
                switch (direction) {
          
            case 1: //�������� �����-�������

                if (FLAMEGPU->getVariable<int>("agent_state")==3 && 
                    FLAMEGPU->getVariable<float>("x") >= x_min + L/2 &&
					FLAMEGPU->getVariable<float>("y") >= y_min + L/2 - w &&
                    FLAMEGPU->getVariable<float>("y") <= y_min + L/2)
                    FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                if( (FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4) )
                {
                    
                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity"));

                   
                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") *sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                                 (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 - w &&
                             FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                                 (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2)
                    {
                        
                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                            FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                            FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));
                         
                    }
                  
                    double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        y_potentional2 >= y_min + L/2 - w &&
                        y_potentional2 <= y_min + L/2) 
                    {
                        FLAMEGPU->setVariable<float>("x", x_potentional2);
                        FLAMEGPU->setVariable<float>("y", y_potentional2);
                    }
                    

                }

                if(FLAMEGPU->getVariable<int>("agent_state") == 3)
                    {
                    alpha = 1 * FLAMEGPU->getVariable<float>("velocity") / distance;

                    double x_potential1 = FLAMEGPU->getVariable<float>("x") +  cos(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                                                                (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                    double y_potential1 = FLAMEGPU->getVariable<float>("y") + sin(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                                                                (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                    {
                    FLAMEGPU->setVariable<float>("x", x_center + distance * cos(alpha + beta) );
                    FLAMEGPU->setVariable<float>("y", y_center + distance * sin(alpha + beta) );
                    }
                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if(pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) >= R1 &&
                            pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential1);
                        FLAMEGPU->setVariable<float>("y", y_potential1);
                    }

                    double x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));
                    
                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential2);
                        FLAMEGPU->setVariable<float>("y", y_potential2);
                    }


                    }

                break;
            
            case 2: // �������� ������-������
                if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
                    FLAMEGPU->getVariable<float>("x") <= x_min + L/2 &&
					FLAMEGPU->getVariable<float>("y") >= y_min + L / 2  &&
                    FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 + w)
                    FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                {

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("velocity"));


                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 &&

                        FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + w)
                    {

                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                            FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                            FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                    }

                    double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        y_potentional2 >= y_min + L / 2 &&
                        y_potentional2 <= y_min + L / 2 + w)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potentional2);
                        FLAMEGPU->setVariable<float>("y", y_potentional2);
                    }


                }

                if (FLAMEGPU->getVariable<int>("agent_state") == 3)
                {
                    alpha = 1 * FLAMEGPU->getVariable<float>("velocity") / distance;

                    double x_potential1 = FLAMEGPU->getVariable<float>("x") + cos(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                    double y_potential1 = FLAMEGPU->getVariable<float>("y") + sin(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                    {
                        FLAMEGPU->setVariable<float>("x", x_center + distance * cos(alpha + beta));
                        FLAMEGPU->setVariable<float>("y", y_center + distance * sin(alpha + beta));
                    }
                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential1);
                        FLAMEGPU->setVariable<float>("y", y_potential1);
                    }

                    double x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential2);
                        FLAMEGPU->setVariable<float>("y", y_potential2);
                    }
                }

               break;
            
            case 3: // �������� �����-�����
                if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
					FLAMEGPU->getVariable<float>("x") >= x_min + L / 2  &&
                    FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w &&
                    FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                    FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                {

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity"));


                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 &&
                             FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 + w)
                    {

                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                            FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                            FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                    }

                    double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        x_potentional2 >= x_min + L / 2 &&
                        x_potentional2 <= x_min + L / 2 + w)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potentional2);
                        FLAMEGPU->setVariable<float>("y", y_potentional2);
                    }


                }

                if (FLAMEGPU->getVariable<int>("agent_state") == 3)
                {
                    alpha = 1 * FLAMEGPU->getVariable<float>("velocity") / distance;

                    double x_potential1 = FLAMEGPU->getVariable<float>("x") + cos(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                    double y_potential1 = FLAMEGPU->getVariable<float>("y") + sin(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                    {
                        FLAMEGPU->setVariable<float>("x", x_center + distance * cos(alpha + beta));
                        FLAMEGPU->setVariable<float>("y", y_center + distance * sin(alpha + beta));
                    }
                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential1);
                        FLAMEGPU->setVariable<float>("y", y_potential1);
                    }

                    double x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential2);
                        FLAMEGPU->setVariable<float>("y", y_potential2);
                    }
                }

                break;
            case 4: // �������� ������-����
                if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
					FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 - w &&
                    FLAMEGPU->getVariable<float>("x") <= x_min + L / 2  &&
                    FLAMEGPU->getVariable<float>("y") <= y_min + L / 2)
                    FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                {

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") - FLAMEGPU->getVariable<float>("velocity"));


                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 - w &&
                        FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2)
                    {

                        FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                            FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                        FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                            FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                            (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                    }

                    double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        x_potentional2 >= x_min + L / 2 - w &&
                        x_potentional2 <= x_min + L / 2)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potentional2);
                        FLAMEGPU->setVariable<float>("y", y_potentional2);
                    }


                }

                if (FLAMEGPU->getVariable<int>("agent_state") == 3)
                {
                    alpha = 1 * FLAMEGPU->getVariable<float>("velocity") / distance;

                    double x_potential1 = FLAMEGPU->getVariable<float>("x") + cos(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                    double y_potential1 = FLAMEGPU->getVariable<float>("y") + sin(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                        (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                    if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                    {
                        FLAMEGPU->setVariable<float>("x", x_center + distance * cos(alpha + beta));
                        FLAMEGPU->setVariable<float>("y", y_center + distance * sin(alpha + beta));
                    }
                    // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                    else if (pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential1);
                        FLAMEGPU->setVariable<float>("y", y_potential1);
                    }

                    double x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                    double y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                    // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                    if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R3)
                    {
                        FLAMEGPU->setVariable<float>("x", x_potential2);
                        FLAMEGPU->setVariable<float>("y", y_potential2);
                    }
                }
                break;
            }
           }

           if (DRN == 2 || DRN == 5 || DRN == 4 || DRN == 6) 
           {
               switch (direction) {

               case 1: //�������� �����-�������

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN != 5 || DRN != 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 - w &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   double prob1 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 - 3 * w &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 - 2 * w && prob1 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   double prob2 = FLAMEGPU->random.uniform<float>();

                       if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                           FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 &&
                           FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 - w &&
                           FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 && prob2 < 0.1)
                           FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   double prob3 = FLAMEGPU->random.uniform<float>();

                       if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                           FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 &&
                           FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 + w &&
                           FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 + 2 * w && prob3 < 0.1)
                           FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity"));


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       else if (
                           
                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 - w&&
                           FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 ) ||

                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 - 3 * w &&
                               FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 - 2 * w) ||

                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 +  w &&
                               FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + 2 * w)

                               )
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if  (
                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                           y_potentional2 >= y_min + L / 2 - w &&
                           y_potentional2 <= y_min + L / 2) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               y_potentional2 >= y_min + L / 2 - 3* w &&
                               y_potentional2 <= y_min + L / 2 - 2* w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               y_potentional2 >= y_min + L / 2 + w &&
                               y_potentional2 <= y_min + L / 2 + 2*w) 
                           )

                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   

                   break;

               case 2: // �������� ������-������
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN!=5 || DRN!=6) &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 + w)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob1 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN != 5 || DRN != 6) &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 - 2 *w &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 - w && prob1< 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����


                   prob2 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN != 5 || DRN != 6) &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2  &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 + w && prob2 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob3 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN != 5 || DRN != 6) &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 + 2*w &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2 + 3*w && prob3 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����
                   

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("velocity"));


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       else if (
                           
                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 &&
                           FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + w) ||

                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 - 2 *w &&
                               FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 - w) ||

                           (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 + 2 * w &&
                               FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + 3 * w)
                               
                               )
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if  (
                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                           y_potentional2 >= y_min + L / 2 &&
                           y_potentional2 <= y_min + L / 2 + w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               y_potentional2 >= y_min + L / 2 - 2 *w &&
                               y_potentional2 <= y_min + L / 2 - w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               y_potentional2 >= y_min + L / 2 + 2 * w &&
                               y_potentional2 <= y_min + L / 2 + 3 * w)
                           )

                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 3: // �������� �����-�����
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN!=5 || DRN!=6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2  &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob1 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 - 2*w &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 - w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob1 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����


                   prob2 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2  &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob2 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob3 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 + 2*w &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + 3*w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob3 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����


                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity"));


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       else if (
                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 &&
                           FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 + w) ||

                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 - 2 * w &&
                               FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 - w) ||

                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 + 2 * w &&
                               FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 + 3 * w) 
                              )

                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if (
                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                           x_potentional2 >= x_min + L / 2 &&
                           x_potentional2 <= x_min + L / 2 + w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               x_potentional2 >= x_min + L / 2 - 2 * w &&
                               x_potentional2 <= x_min + L / 2 -  w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               x_potentional2 >= x_min + L / 2 + 2 * w &&
                               x_potentional2 <= x_min + L / 2 + 3 * w)
                           )
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 4: // �������� ������-����
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN != 5 || DRN != 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 - w &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") <= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob1 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 - 3 * w &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 - 2 * w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob1 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����


                   prob2 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 - 2&&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob2 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   prob3 = FLAMEGPU->random.uniform<float>();

                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 && (DRN == 5 || DRN == 6) &&
                       FLAMEGPU->getVariable<float>("x") >= x_min + L / 2 + w &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + 2 * w &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2 && prob3 < 0.1)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") - FLAMEGPU->getVariable<float>("velocity"));


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       else if (
                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 - w &&
                           FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2) ||

                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 - 3 * w &&
                           FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 - 2 * w) ||

                           (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 + w &&
                           FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 + 2 * w)
                               )

                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if  (
                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                           x_potentional2 >= x_min + L / 2 - w &&
                           x_potentional2 <= x_min + L / 2) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               x_potentional2 >= x_min + L / 2 - 3 * w &&
                               x_potentional2 <= x_min + L / 2 - 2 * w) ||

                           (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                               x_potentional2 >= x_min + L / 2 + w &&
                               x_potentional2 <= x_min + L / 2 + 2 * w)
                           )
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 5: // �������� �-�
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                       {
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity")*cos(45 * M_PI / 180));
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity")*sin(45 * M_PI / 180));
                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") +
                           FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") +
                           FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));


                       double x2 = x_min + L - (double)w / sin(45 * M_PI / 180);
                       double y2 = y_min + L;
                       double x1 = x_min - (double)w / sin(45 * M_PI / 180);
                       double y1 = y_min;


                       double x4 = x_min + L;
                       double y4 = y_min + L;
                       double x3 = x_min;
                       double y3 = y_min;

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       double sign_road1 =  (x2 - x1)*(y_potentional2 - y1) - (y2-y1)*(x_potentional2 - x1);
                       double sign_road2 =  (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);
                       

                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &  // ���� ����������
                           sign_road1 < 0 && sign_road2 > 0)
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);

                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &
                           sign_road1 < 0 && sign_road2 > 0 )
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 6: // �������� �-�
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                       {
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("velocity") * cos(45 * M_PI / 180));
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") - FLAMEGPU->getVariable<float>("velocity") * sin(45 * M_PI / 180));
                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") -
                           FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) -
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") -
                           FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) -
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));


                       double x2 = x_min + L;
                       double y2 = y_min + L;
                       double x1 = x_min;
                       double y1 = y_min;

                       double x4 = x_min + L + (double)w / sin(45 * M_PI / 180);
                       double y4 = y_min + L;
                       double x3 = x_min + (double)w / sin(45 * M_PI / 180);
                       double y3 = y_min;

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       double sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       double sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &  // ���� �����������
                           sign_road1 < 0 && sign_road2 > 0)
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") -
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) -
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") -
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) -
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       x_potentional2 = FLAMEGPU->getVariable<float>("x") - (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       y_potentional2 = FLAMEGPU->getVariable<float>("y") - (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);

                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &
                           sign_road1 < 0 && sign_road2 > 0)
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 7: // �������� �-�
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                       {
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(-45 * M_PI / 180));
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(-45 * M_PI / 180));
                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") +
                           FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") +
                           FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));


                       double x2 = x_min + L;
                       double y2 = y_min;
                       double x1 = x_min;
                       double y1 = y_min + L;

                       double x4 = x_min + L - (double)w / sin(45 * M_PI / 180);
                       double y4 = y_min;
                       double x3 = x_min - (double)w / sin(45 * M_PI / 180);
                       double y3 = y_min + L;

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       double sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       double sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &  // ���� ����������
                           sign_road1 < 0 && sign_road2 > 0)
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);

                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &
                           sign_road1 < 0 && sign_road2 > 0)
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;

               case 8: // �������� �-�
                   if (FLAMEGPU->getVariable<int>("agent_state") == 3 &&
                       FLAMEGPU->getVariable<float>("x") <= x_min + L / 2 + w / 2 &&
                       FLAMEGPU->getVariable<float>("y") >= y_min + L / 2)
                       FLAMEGPU->setVariable<int>("agent_state", 4); // ����� � �����

                   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
                   {

                       if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
                       {
                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("velocity") * cos(45 * M_PI / 180));
                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(45 * M_PI / 180));
                       }

                       double x_potentional2 = FLAMEGPU->getVariable<float>("x") -
                           FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) -
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                       double y_potentional2 = FLAMEGPU->getVariable<float>("y") +
                           FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                           (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));


                       double x2 = x_min + L + (double)w / sin(45 * M_PI / 180);
                       double y2 = y_min;
                       double x1 = x_min  + (double)w / sin(45 * M_PI / 180);
                       double y1 = y_min + L;

                       double x4 = x_min + L;
                       double y4 = y_min;
                       double x3 = x_min;
                       double y3 = y_min + L;

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       double sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       double sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);


                       // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &  // ���� ����������
                           sign_road1 < 0 && sign_road2 > 0)
                       {

                           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") -
                               FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) -
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

                           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
                               FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
                               (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

                       }

                       x_potentional2 = FLAMEGPU->getVariable<float>("x") - (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                       y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                       // ���� sign_road > 0, �� ���� ������, ��� sign_road < 0, �� ���� ������������ ������ 
                       sign_road1 = (x2 - x1) * (y_potentional2 - y1) - (y2 - y1) * (x_potentional2 - x1);
                       sign_road2 = (x4 - x3) * (y_potentional2 - y3) - (y4 - y3) * (x_potentional2 - x3);

                       // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
                       if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &
                           sign_road1 < 0 && sign_road2 > 0)
                       {
                           FLAMEGPU->setVariable<float>("x", x_potentional2);
                           FLAMEGPU->setVariable<float>("y", y_potentional2);
                       }


                   }

                   break;
               }

               if (FLAMEGPU->getVariable<int>("agent_state") == 3)
               {
                   alpha =   1 * FLAMEGPU->getVariable<float>("velocity") / distance;

                  
                   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance") &&
                          (pow(pow(distance * cos(-alpha + beta), 2) + pow(distance * sin(-alpha + beta), 2), 0.5) >= R1 &&
                           pow(pow(distance * cos(-alpha + beta), 2) + pow(distance * sin(-alpha + beta), 2), 0.5) <= R3)) // ��� �����������
                   {
                       FLAMEGPU->setVariable<float>("x", x_center + distance * cos(-alpha + beta));
                       FLAMEGPU->setVariable<float>("y", y_center + distance * sin(-alpha + beta));
                   }

                   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance") &&
                       (pow(pow(distance * cos(alpha + beta), 2) + pow(distance * sin(alpha + beta), 2), 0.5) >= R3 &&
                        pow(pow(distance * cos(alpha + beta), 2) + pow(distance * sin(alpha + beta), 2), 0.5) <= R5)) // ��� �����������
                   {
                       FLAMEGPU->setVariable<float>("x", x_center + distance * cos(alpha + beta));
                       FLAMEGPU->setVariable<float>("y", y_center + distance * sin(alpha + beta));
                   }

                   double x_potential1 = FLAMEGPU->getVariable<float>("x") + cos(-alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                       (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                   double y_potential1 = FLAMEGPU->getVariable<float>("y") + sin(-alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) +
                       (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                   // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
                   if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &&
                       (pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) >= R1 &&
                        pow(pow(x_potential1 - x_center, 2) + pow(y_potential1 - y_center, 2), 0.5) <= R3))
                   {
                       FLAMEGPU->setVariable<float>("x", x_potential1);
                       FLAMEGPU->setVariable<float>("y", y_potential1);
                   }

                   double x_potential2 = FLAMEGPU->getVariable<float>("x") + cos(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) -
                       (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma"));

                   double y_potential2 = FLAMEGPU->getVariable<float>("y") + sin(alpha + beta + sign * FLAMEGPU->getVariable<float>("omega")) -
                       (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma"));

                   if (FLAMEGPU->getVariable<float>("neighbour_distance") <= FLAMEGPU->getVariable<float>("threshold_distance") &&
                       (pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R3 &&
                       pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R5))

                   {
                       FLAMEGPU->setVariable<float>("x", x_potential2);
                       FLAMEGPU->setVariable<float>("y", y_potential2);
                   }

                   x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                   y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                       (pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R1 &&
                           pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R3))

                   {
                       FLAMEGPU->setVariable<float>("x", x_potential2);
                       FLAMEGPU->setVariable<float>("y", y_potential2);
                   }

                   x_potential2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
                   y_potential2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));

                   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ���� ��������� �������
                   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
                       (pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) >= R3 &&
                           pow(pow(x_potential2 - x_center, 2) + pow(y_potential2 - y_center, 2), 0.5) <= R5))
                   {
                       FLAMEGPU->setVariable<float>("x", x_potential2);
                       FLAMEGPU->setVariable<float>("y", y_potential2);
                   }
               }
           }

		   if (DRN == 3) // ������ ���
		   {
			   switch (direction) {

			   case 1: //�������� �����-�������

				   
				   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
				   {

					   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity"));


					   // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
					   else if (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 &&
						   FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + w)
					   {

						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
							   FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
							   FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

					   }

					   double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
					   double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


					   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
					   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
						   y_potentional2 >= y_min + L / 2 &&
						   y_potentional2 <= y_min + L / 2)
					   {
						   FLAMEGPU->setVariable<float>("x", x_potentional2);
						   FLAMEGPU->setVariable<float>("y", y_potentional2);
					   }


				   }

				   break;

			   case 2: // �������� ������-������
				
				   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
				   {

					   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("velocity"));


					   // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
					   else if (FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) >= y_min + L / 2 &&

						   FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")) <= y_min + L / 2 + w)
					   {

						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
							   FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
							   FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

					   }

					   double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
					   double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


					   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
					   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
						   y_potentional2 >= y_min + L / 2 &&
						   y_potentional2 <= y_min + L / 2)
					   {
						   FLAMEGPU->setVariable<float>("x", x_potentional2);
						   FLAMEGPU->setVariable<float>("y", y_potentional2);
					   }


				   }

				   break;

			   case 3: // �������� �����-�����
				   
				   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
				   {

					   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("velocity"));


					   // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
					   else if (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 &&
						   FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2 + w)
					   {

						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
							   FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
							   FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

					   }

					   double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
					   double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


					   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
					   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
						   x_potentional2 >= x_min + L / 2 &&
						   x_potentional2 <= x_min + L / 2 + w)
					   {
						   FLAMEGPU->setVariable<float>("x", x_potentional2);
						   FLAMEGPU->setVariable<float>("y", y_potentional2);
					   }


				   }

				   break;
			   case 4: // �������� ������-����
				  
				   if ((FLAMEGPU->getVariable<int>("agent_state") == 2 || FLAMEGPU->getVariable<int>("agent_state") == 4))
				   {

					   if (FLAMEGPU->getVariable<float>("neighbour_distance") > FLAMEGPU->getVariable<float>("threshold_distance")) // ��� �����������
						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") - FLAMEGPU->getVariable<float>("velocity"));


					   // ����� ����������� ������� ���������� ������-�� ��� ������� ������� �� ��������� ������
					   else if (FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) >= x_min + L / 2 - w &&
						   FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
						   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")) <= x_min + L / 2)
					   {

						   FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") +
							   FLAMEGPU->getVariable<float>("velocity") * cos(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("gamma")));

						   FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") +
							   FLAMEGPU->getVariable<float>("velocity") * sin(sign * FLAMEGPU->getVariable<float>("omega")) +
							   (c1 / FLAMEGPU->getVariable<float>("neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("gamma")));

					   }

					   double x_potentional2 = FLAMEGPU->getVariable<float>("x") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * cos(FLAMEGPU->getVariable<float>("other_gamma"));
					   double y_potentional2 = FLAMEGPU->getVariable<float>("y") + (c1 / FLAMEGPU->getVariable<float>("other_neighbour_distance")) * sin(FLAMEGPU->getVariable<float>("other_gamma"));


					   // �������� � ��������������� �� ���������� ������-�� �����������  ��� ������� ������� �� ��������� ������
					   if (FLAMEGPU->getVariable<float>("other_neighbour_distance") < FLAMEGPU->getVariable<float>("other_threshold_distance") &&
						   x_potentional2 >= x_min + L / 2 - w &&
						   x_potentional2 <= x_min + L / 2)
					   {
						   FLAMEGPU->setVariable<float>("x", x_potentional2);
						   FLAMEGPU->setVariable<float>("y", y_potentional2);
					   }


				   }

				   break;
			   }
		   }
		   
       }

       if (FLAMEGPU->getVariable<int>("agent_state") == 1)
       {
           FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") );
           FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
       }

        
        double rot = atan2((FLAMEGPU->getVariable<float>("y") - FLAMEGPU->getVariable<float>("y_past")), (FLAMEGPU->getVariable<float>("x") - FLAMEGPU->getVariable<float>("x_past")));
        if(FLAMEGPU->getVariable<int>("agent_state") != 1)
          FLAMEGPU->setVariable<float>("rotation", rot);
        else
          FLAMEGPU->setVariable<float>("rotation", FLAMEGPU->getVariable<float>("rotation_past") );

        //���������� �������� ��������� ������-��
        FLAMEGPU->setVariable<float>("x_past", FLAMEGPU->getVariable<float>("x") );
        FLAMEGPU->setVariable<float>("y_past", FLAMEGPU->getVariable<float>("y") );
        FLAMEGPU->setVariable<float>("rotation_past", FLAMEGPU->getVariable<float>("rotation"));


    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(update_agent_state, flamegpu::MessageNone, flamegpu::MessageSpatial2D)
{
    FLAMEGPU->setVariable<float>("threshold_distance", 0);
    FLAMEGPU->setVariable<float>("other_threshold_distance", 0);

    int agent_type = FLAMEGPU->getVariable<int>("agent_type");
    int agent_state = FLAMEGPU->getVariable<int>("agent_state");

    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");

    int x_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_x");
    int x_max = FLAMEGPU->environment.getProperty<unsigned int>("L") + FLAMEGPU->environment.getProperty<unsigned int>("indent_x");
    int y_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_y");
    int y_max = FLAMEGPU->environment.getProperty<unsigned int>("L") + FLAMEGPU->environment.getProperty<unsigned int>("indent_y");

    auto traffic = FLAMEGPU->environment.getMacroProperty<uint32_t>("Traffic");

    //������� �������-�� �� ���
    if (agent_type == 1 && x > x_max || agent_type == 2 && x < x_min || agent_type == 3 && y > y_max || agent_type == 4 && y < y_min ||
        agent_type == 5 && y > y_max || agent_type == 6 && y < y_min || agent_type == 8 && y > y_max || agent_type == 7 && y < y_min)
        {
        unsigned int vehicles_count = traffic++;
        //printf("%u \n", vehicles_count); // ������� ��������� ������
        return flamegpu::DEAD;
        }
    else
    {
        //�������� ������ � �������������� ������-�� � ��� ����������
        FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->message_out.setVariable<float>("Ra", FLAMEGPU->getVariable<float>("Ra"));
        FLAMEGPU->message_out.setVariable<int>("agent_type", FLAMEGPU->getVariable<int>("agent_type"));
        FLAMEGPU->message_out.setVariable<int>("agent_state", FLAMEGPU->getVariable<int>("agent_state"));
        
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"));

        return flamegpu::ALIVE;
    }
}


//������ ��������� ������ ������ ������ � ��������� ������� ������� ������������ ������ � ��� ��������
FLAMEGPU_AGENT_FUNCTION(density_estimation, flamegpu::MessageSpatial2D, flamegpu::MessageNone) 
{
    const float RADIUS = FLAMEGPU->message_in.radius();
    float density = 0.0;
    float separation = 0.0;

    FLAMEGPU->setVariable<float>("neighbour_distance", 10000);
    FLAMEGPU->setVariable<float>("other_neighbour_distance", 10000);

    // Get this agent's x, y, z variables
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    int flag = 0;
    

    // For each message in the message list which was output by a nearby agent
    for (const auto& message : FLAMEGPU->message_in(x1, y1)) {
        const float x2 = message.getVariable<float>("x");
        const float y2 = message.getVariable<float>("y");

        // Calculate the distance to check the message is in range
        float x21 = x2 - x1;
        float y21 = y2 - y1;

        //��������� ���� ����������� (������) �� ��������� ������, � ��������
        float dir_angle = (180/M_PI) * atan2(y21, x21);
        

        //printf("%f\n", (180 / M_PI)*atan2(y21, x21));

        if (message.getVariable<int>("id") != FLAMEGPU->getVariable<int>("id"))
        {
            separation = sqrt(x21 * x21 + y21 * y21);
            if (separation < INTERACTION_RADIUS) {
                // Process the message
                density++;
                //int idFromMessage = message.getVariable<int>("id");
                flag = 1;
            }

            if (separation == 0)
                separation = 1000000;


            if (separation < FLAMEGPU->getVariable<float>("neighbour_distance") && abs(dir_angle) <= 0.001)
            {
                //������ ���������� �� ���������� ������ � ������ ���������� ���������� ������, �������������� ������� �� ���� ��������

                FLAMEGPU->setVariable<float>("neighbour_distance", separation);
                FLAMEGPU->setVariable<float>("threshold_distance", FLAMEGPU->getVariable<float>("Ra") + message.getVariable<float>("Ra")); // ��������� ���������� ����� ��������: ����� �������� ������ ����������� �������� �������
                FLAMEGPU->setVariable<int>("id_neighbour", message.getVariable<int>("id"));
                FLAMEGPU->setVariable<float>("x_neighbour", message.getVariable<float>("x")); // ���������� ���������� ������
                FLAMEGPU->setVariable<float>("y_neighbour", message.getVariable<float>("y"));
                FLAMEGPU->setVariable<float>("omega", (M_PI / 4) +    (atan2(message.getVariable<float>("y") - FLAMEGPU->getVariable<float>("y") + (FLAMEGPU->getVariable<float>("Ra") + message.getVariable<float>("Ra")) * sin(M_PI / 4),
                                                                         message.getVariable<float>("x") - FLAMEGPU->getVariable<float>("x") + (FLAMEGPU->getVariable<float>("Ra") + message.getVariable<float>("Ra")) * cos(M_PI / 4))));
                
               // FLAMEGPU->setVariable<float>("omega", (M_PI / 4));

                FLAMEGPU->setVariable<float>("gamma", M_PI + (atan2(message.getVariable<float>("y") - FLAMEGPU->getVariable<float>("y"),
                                                                    message.getVariable<float>("x") - FLAMEGPU->getVariable<float>("x"))));
                FLAMEGPU->setVariable<float>("neighbour_angle", dir_angle); // ���� ����������� �� ���������� ������

            }

            if (separation < FLAMEGPU->getVariable<float>("other_neighbour_distance") && abs(dir_angle) > 0.001)
            {
                //������ ���������� �� ���������� ������ � ������ ���������� ���������� ������

                FLAMEGPU->setVariable<float>("other_neighbour_distance", separation);
                FLAMEGPU->setVariable<float>("other_threshold_distance", FLAMEGPU->getVariable<float>("Ra") + message.getVariable<float>("Ra")); // ��������� ���������� ����� ��������: ����� �������� ������ ����������� �������� �������
                FLAMEGPU->setVariable<float>("other_gamma", M_PI + (atan2(message.getVariable<float>("y") - FLAMEGPU->getVariable<float>("y"),
                                                                          message.getVariable<float>("x") - FLAMEGPU->getVariable<float>("x"))));
                FLAMEGPU->setVariable<float>("neighbour_angle", dir_angle); // ���� ����������� �� ���������� ������
            }

            int agent_type = FLAMEGPU->getVariable<int>("agent_type");
            int agent_state = FLAMEGPU->getVariable<int>("agent_state");

            float x = FLAMEGPU->getVariable<float>("x");
            float y = FLAMEGPU->getVariable<float>("y");

            int x_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_x");
            int x_max = FLAMEGPU->environment.getProperty<unsigned int>("L") + FLAMEGPU->environment.getProperty<unsigned int>("indent_x");
            int y_min = FLAMEGPU->environment.getProperty<unsigned int>("indent_y");
            int y_max = FLAMEGPU->environment.getProperty<unsigned int>("L") + FLAMEGPU->environment.getProperty<unsigned int>("indent_y");
        
            if (separation < 1 && message.getVariable<int>("agent_state") != 1 &&
                !(agent_type == 1 && x > x_max || agent_type == 2 && x < x_min || agent_type == 3 && y > y_max || agent_type == 4 && y < y_min) &&
                  FLAMEGPU->getVariable<float>("neighbour_distance") < 10000 )
                { 
                FLAMEGPU->setVariable<int>("agent_state", 1); // ��������� ��������

               
                printf("%f, %f, %f, %f, %f  TRAFFIC ACCIDENT \n", separation, 
                                                                  FLAMEGPU->getVariable<float>("neighbour_distance"), 
                                                                  FLAMEGPU->getVariable<float>("threshold_distance"),
                                                                  FLAMEGPU->getVariable<float>("omega"),
                                                                  FLAMEGPU->getVariable<float>("gamma") );
              

                }


        }
    }

   
    //��������� �������� ��������� ������ ������ � �������� ������� ��� ������� ������������

    FLAMEGPU->setVariable<float>("density", density);

    double gamma = 1.0;
    if (FLAMEGPU->getVariable<int>("agent_class") == 2)
        gamma = 4; // ���
    else if (FLAMEGPU->getVariable<int>("agent_class") == 1)
        gamma = 0.8; // ���


    if (FLAMEGPU->getVariable<float>("density") <= 1)
        FLAMEGPU->setVariable<float>("Ra", PERSONAL_RADIUS);
    if (FLAMEGPU->getVariable<float>("density") > 1 && FLAMEGPU->getVariable<float>("density") < 20 && FLAMEGPU->getVariable<int>("agent_state") != 1)
        FLAMEGPU->setVariable<float>("Ra", new_radius(FLAMEGPU->getVariable<float>("density"), 1) );
    if (FLAMEGPU->getVariable<float>("density") >= 20 && FLAMEGPU->getVariable<int>("agent_state") != 1)
        FLAMEGPU->setVariable<float>("Ra", new_radius(FLAMEGPU->getVariable<float>("density"), gamma) );


    //printf("%f\n", FLAMEGPU->getVariable<float>("Ra") );

    return flamegpu::ALIVE;
}


int main(int argc, const char** argv) {
flamegpu::ModelDescription model("Transportation");

const float RADIUS = 150.0f;


flamegpu::MessageSpatial2D::Description& message_distance = model.newMessage<flamegpu::MessageSpatial2D>("location_agent");
{
    message_distance.newVariable<int>("id");
    message_distance.newVariable<float>("Ra");
    message_distance.newVariable<int>("agent_type");
    message_distance.newVariable<int>("agent_state");

    message_distance.setRadius(RADIUS);
    message_distance.setMin(0, 0);
    message_distance.setMax(window_width, window_height);
}

flamegpu::AgentDescription& agent = model.newAgent("agent-vehicles");
{
    agent.newVariable<int>("id"); // ID
    agent.newVariable<float>("x"); // ���������� ������ � ���������� ������� ���������
    agent.newVariable<float>("y");
    agent.newVariable<float>("x_past"); // ��������� �������� ��������� ������-��
    agent.newVariable<float>("y_past");
    agent.newVariable<float>("rotation_past");
    agent.newVariable<int>("agent_class"); // 1 - ���, 2 - ���
    agent.newVariable<int>("agent_state"); // 0 - ����������, 1 - ���������
    agent.newVariable<int>("agent_type"); // 1 - �����-�������, 2-������-������, 3-�����-�����, 4-������-����
    agent.newVariable<float>("Ra"); // ������ ������� ������������ ������
    agent.newVariable<float>("rotation"); // ���� �������� ��
    agent.newVariable<float>("velocity"); // �������� ������
    agent.newVariable<float>("density"); // ��������� ��������� ������ ������ ������
    agent.newVariable<float>("alpha"); // ����, ������������ ����������� �������� ������ (������)
    agent.newVariable<float>("omega"); // ����, ������� ����������� �����������
    agent.newVariable<float>("gamma"); // ����, ������� �� ����������

    agent.newVariable<float>("neighbour_distance"); // ��������� ���������� ������, ������������� ������� �� ���� ��������
    agent.newVariable<float>("threshold_distance");
    agent.newVariable<float>("id_neighbour");
    agent.newVariable<float>("x_neighbour");
    agent.newVariable<float>("y_neighbour");
    agent.newVariable<float>("neighbour_angle");

    agent.newVariable<float>("other_neighbour_distance"); // ��������� ���������� ������
    agent.newVariable<float>("other_threshold_distance");
    agent.newVariable<float>("other_gamma");
        
}

//������� ����������� �������-��
flamegpu::AgentFunctionDescription& agent_fn1_description = agent.newFunction("agent_move", agent_move);

//������� ���������� ��������� �������  
auto& fn_state_update = agent.newFunction("update_agent_state", update_agent_state);  // ���������� ��������� ������
{
    fn_state_update.setMessageOutput("location_agent"); // �������� ������ � �������������� ������-��
    fn_state_update.setAllowAgentDeath(true);
}

// ������� ������ ��������� ��������� ������ � ���������� �� ���������� ������
auto& fn_all_agents = agent.newFunction("density_estimation", density_estimation);
{
    fn_all_agents.setMessageInput("location_agent"); // ��������� ������ � �������������� ������-��
}

model.addInitFunction(init_function);
model.addStepFunction(BasicOutput);
model.addExitCondition(exit_condition); // ����������� ��� ������

{   // Layer #4 ������� �����
    flamegpu::LayerDescription& layer = model.newLayer();
    layer.addHostFunction(agents_data_updating);
}


{   // Layer #1 
    flamegpu::LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(update_agent_state);
}

{   // Layer #2 
    flamegpu::LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(density_estimation);
}

{ // Layer  #3 ������� ������
 flamegpu::LayerDescription& layer = model.newLayer();
 layer.addAgentFunction(agent_move);
}


//�������� �����
flamegpu::EnvironmentDescription& env = model.Environment();
env.newProperty<unsigned int>("DRN", 1);        // ��� �������� �������� ����
env.newProperty<unsigned int>("intensity_of_UGVs", 5); // ������������� �������� ��� � ��� � ������� ���������� �������
env.newProperty<unsigned int>("intensity_of_MGVs", 5); // ������������� �������� ��� � ��� � ������� ���������� �������
env.newProperty<float>("velocity_of_UGVs", 10); // ������� �������� ��� 100 ��/�
env.newProperty<float>("velocity_of_MGVs", 10); // ������� �������� ��� 100 ��/�
env.newProperty<unsigned int>("intensity_of_abnormal_MGVs", 10); // ������������� �������� ���, � ���������� ���������� � ���
env.newProperty<unsigned int>("frequency", 10); // ������� �������� ��� � ��� � ��� (������ N ������)

env.newProperty<unsigned int>("L", 1000); // ����� �����
env.newProperty<unsigned int>("w", 100); // ������ �����
env.newProperty<unsigned int>("N_nodes", 2); // ���������� ������� ���
env.newProperty<unsigned int>("R1", 300); // ���������� ������ ���� ��������� ��������

env.newProperty<unsigned int>("indent_x", 10); // ������� ��� ���
env.newProperty<unsigned int>("indent_y", 10);

env.newMacroProperty<uint32_t>("Traffic"); // ������� ��������� ������ 

/*
* Create Model Runner
*/

flamegpu::CUDASimulation cuda_model(model);
cuda_model.initialise(1, argv);
cuda_model.SimulationConfig().steps = TIME_STOP;
flamegpu::AgentVector population1(model.Agent("agent-vehicles"), 1000);

//��������� ������ ������������ ������ � �������������
if (VIS_MODE == 1)
{
    initVisualisation();
    glutTimerFunc(1, timer, 0);
    cuda_model.SimulationConfig().steps = TIME_STOP;
    std::thread first([&cuda_model]() { cuda_model.simulate(); });
    runVisualisation(); //������ ������������
    first.join();
}


//������������ ������������
if (VIS_MODE == 3)
{
    for (int intensity_of_UGVs = 1; intensity_of_UGVs <= 10; intensity_of_UGVs++)
    {
        for (int velocity_of_UGVs = 1; velocity_of_UGVs <= 10; velocity_of_UGVs++)
        {
            flamegpu::CUDASimulation cuda_model(model);
            cuda_model.initialise(1, argv);

            cuda_model.SimulationConfig().steps = TIME_STOP;
            
            par1 = intensity_of_UGVs;
            par2 = velocity_of_UGVs;
            flamegpu::AgentVector population1(model.Agent("agent-vehicles"), 1000);
            std::thread second([&cuda_model]() { cuda_model.simulate(); });
            second.join();

            //������� �������, ����������� � ������
            int obj = objective.load();

            if (out.is_open())
            {
                out << obj <<
                    ";" << intensity_of_UGVs <<
                    ";" << velocity_of_UGVs << std::endl;
            }

            flamegpu::util::cleanup();

        }
    }
}



out.close();


    return 0;
}
