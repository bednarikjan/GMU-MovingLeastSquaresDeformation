/*
 *  SchaeferMLS.cpp
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 12/28/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "std.h"
using namespace cv;

#include "SchaeferMLS.h"
#include "CurveSignature.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif //WIN32

SchaeferMLS<double> smls(true);
Mat visualized_curve;
vector<Point> target_curve;
const int mls_def_type = 2;

vector<Point2d> curve;
vector<int> keyPointIndices;

double GetTime(void)
{
#if _WIN32  															/* toto jede na Windows */
    static int initialized = 0;
    static LARGE_INTEGER frequency;
    LARGE_INTEGER value;

    if (!initialized) {                         							/* prvni volani */
        initialized = 1;
        if (QueryPerformanceFrequency(&frequency) == 0) {                   /* pokud hi-res pocitadlo neni podporovano */
            //assert(0 && "HiRes timer is not available.");
            exit(-1);
        }
    }

    //assert(QueryPerformanceCounter(&value) != 0 && "This should never happen.");  /* osetreni chyby */
    QueryPerformanceCounter(&value);
    return (double)value.QuadPart / (double)frequency.QuadPart;  			/* vrat hodnotu v sekundach */

#else                                         							/* toto jede na Linux/Unixovych systemech */
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == -1) {        							/* vezmi cas */
        //assert(0 && "gettimeofday does not work.");  						/* osetri chyby */
        exit(-2);
    }
    return (double)tv.tv_sec + (double)tv.tv_usec/1000000.;  				/* vrat cas v sekundach */
#endif
}

void MLSUpdate() {
	static int framenum = 0;
	
	if (mls_def_type == 0) {
		smls.UpdateAffine();
	} else if (mls_def_type == 1) {
		smls.UpdateSimilarity();
	} else {
		smls.UpdateRigid();
	}

    visualized_curve.setTo(0);
    smls.Draw(visualized_curve);
    imshow("MLS", visualized_curve);
    waitKey(1);
}	

void onMouse( int event, int x, int y, int flags, void* )
{
	static Point2d start_touch(-1,-1);
	static Point2d last_touch(-1,-1);
	static int touch_control_point = -1;
    static bool lButtonPressed = false;
	
	Point2d touch(x,y);
    if( event == CV_EVENT_LBUTTONDOWN ) {
		//		cout << "mouse down\n";
        lButtonPressed = true;
		start_touch = last_touch = touch;
		const vector<Point2d>& ctrl_pts = smls.GetDeformedControlPts();
		for (int i=0; i<ctrl_pts.size(); i++) {
			if (norm(touch - ctrl_pts[i]) < 10) {
				//touching point i
				touch_control_point = i;
			}
		}
		if (touch_control_point >= 0) {
			cout << "selected point " << touch_control_point << endl;
		}
	} else if( event == CV_EVENT_LBUTTONUP ) {
		//		cout << "mouse up\n";
		touch_control_point = -1;
        lButtonPressed = false;
    } else if (event == CV_EVENT_MOUSEMOVE && lButtonPressed) {
		if (touch_control_point >= 0) {
//            cout << "mouse drag\n";
			vector<Point2d>& def_ctrl_pts = smls.GetDeformedControlPts();
			def_ctrl_pts[touch_control_point] += touch - last_touch;			
			
            vector<Point2d> shifts;
            shifts.resize(def_ctrl_pts.size(), Point2d(0.0, 0.0));
            shifts[touch_control_point] += touch - last_touch;

            double t0 = GetTime();
            smls.deformCurveOneStep(curve, keyPointIndices, shifts);
            double t1 = GetTime();

            cout << "Time elapsed (host): " << t1 - t0 << " s" << endl;

            curve = smls.GetContourPts();

            last_touch = touch;

            visualized_curve.setTo(0);
            smls.Draw(visualized_curve);
            imshow("MLS", visualized_curve);
            waitKey(1);

//            MLSUpdate();
		}
	}
	
}

void onTrackbar(int, void*)
{
	MLSUpdate();
}

void MLSDeformCurve(const Mat& src, 
					const vector<Point2d>& a_p2d,
					const vector<pair<char,int> >& stringrep
					) 	
{
	//Get extrema as control points
	vector<int> control_pts; 
	for(int i=0;i<stringrep.size();i++) {
		control_pts.push_back(stringrep[i].second);
	}

    keyPointIndices = control_pts;
    vector<Point2d> shifts;
    shifts.resize(control_pts.size(), Point2d(0.0, 0.0));
	
//	smls.Init(a_p2d, control_pts);
//	smls.UpdateRigid();

    smls.deformCurveOneStep(a_p2d, keyPointIndices, shifts);
    curve = smls.GetContourPts();
	
	visualized_curve.create(src.size(),CV_8UC3);
	visualized_curve.setTo(0);
	smls.Draw(visualized_curve);

	namedWindow("MLS");
	setMouseCallback("MLS", onMouse, NULL);
//	createTrackbar("Def. type:", "MLS", &mls_def_type, 2, onTrackbar, NULL);
	imshow("MLS", visualized_curve);
	waitKey();
}

void printPoints(const vector<Point2d>& points)
{
    for(int i = 0; i < points.size(); ++i) {
        cout << i << ": [" << points[i].x << ", " << points[i].y << "]" << endl;
    }
}

void testCompareContours(const Mat& src,
                         const vector<Point2d>& a_p2d,
                         const vector<pair<char,int> >& stringrep
                         )
{
    double offsetValue = 20.0;

    //Get extrema as control points
    vector<int> control_pts;
    for(int i=0;i<stringrep.size();i++) {
        control_pts.push_back(stringrep[i].second);
    }

    vector<Point2d> p;
    for (int i=0; i<control_pts.size(); i++) {
        p.push_back(a_p2d[control_pts[i]]);
    }

    cout << "Control points: " <<  endl;
    printPoints(p);

    // Curve 1 (with intermediate control points position)
    smls.Init(a_p2d, control_pts);
    vector<Point2d>& defCtrlPts1 = smls.GetDeformedControlPts();
    for(int i = 0; i < defCtrlPts1.size(); ++i) {
        Point2d offset(offsetValue * i, offsetValue * i);
        defCtrlPts1[i] += offset;
    }
    MLSUpdate();

    cout << "Control points: " <<  endl;
    printPoints(defCtrlPts1);

    vector<Point2d> contourPointsInterm = smls.GetContourPts();
    smls.Init(contourPointsInterm, control_pts);
    for(int i = 0; i < defCtrlPts1.size(); ++i) {
        Point2d offset(offsetValue * i, offsetValue * i);
        defCtrlPts1[i] += offset;
    }
    MLSUpdate();

    cout << "Control points: " <<  endl;
    printPoints(defCtrlPts1);

    vector<Point2d> outputCurve1 = smls.GetContourPts();
    vector<Point2d> outputCtrlPts1 = smls.GetDeformedControlPts();

    visualized_curve.create(src.size(),CV_8UC3);
    visualized_curve.setTo(0);
    smls.Draw(visualized_curve);

    namedWindow("MLS1");
    imshow("MLS1", visualized_curve);
    waitKey(1);

    // Curve 2 (no intermediate control points position)
    smls.Init(a_p2d, control_pts);
    vector<Point2d>& defCtrlPts2 = smls.GetDeformedControlPts();
    for(int i = 0; i < defCtrlPts2.size(); ++i) {
        Point2d offset(2.0 * offsetValue * i, 2.0 * offsetValue * i);
        defCtrlPts2[i] += offset;
    }
    MLSUpdate();

    vector<Point2d> outputCurve2 = smls.GetContourPts();
    vector<Point2d> outputCtrlPts2 = smls.GetDeformedControlPts();

    visualized_curve.create(src.size(),CV_8UC3);
    visualized_curve.setTo(0);
    smls.Draw(visualized_curve);

    namedWindow("MLS2");
    imshow("MLS2", visualized_curve);
    waitKey();

    // Compare curves
    assert(outputCurve1.size() == outputCurve2.size());
    vector<double> diffs;
    double sum = 0.0;
    double diff;
    for(int i = 0; i < outputCurve1.size(); ++i) {
        diff = norm(outputCurve1[i] - outputCurve2[i]);
        sum += diff;
        diffs.push_back(diff);
    }

    // Compare control points
    assert(outputCtrlPts1.size() == outputCtrlPts2.size());
    sum = 0.0;
    for(int i = 0; i < outputCtrlPts1.size(); ++i) {
        sum += norm(outputCtrlPts1[i] - outputCtrlPts2[i]);
    }
    cout << "Curve 1 and Curve 2 ctrl points difference: " << sum << endl;
}

int main(int argc, char** argv) {
    Mat src1 = imread("img/blob.png");
	if (src1.empty()) {
		cerr << "can't read image" << endl; exit(0);
	}
	vector<Point> a;
	GetCurveForImage(src1, a, false);
	
	//move curve a bit to the middle, and scale up
	cv::transform(a,a,getRotationMatrix2D(Point2f(0,0),0,1.3));
//	Mat tmp_curve_m(a); tmp_curve_m += Scalar(100,95);
	
    vector<Point2d> a_p2d, a_p2d_smoothed;
	ConvertCurve(a, a_p2d);

	//Get curvature extrema points
	vector<pair<char,int> > stringrep = CurvatureExtrema(a_p2d, a_p2d_smoothed,0.05,4.0);

    // Print info
    cout << "Curvature points: " << a_p2d.size() << endl;

	//Start interactive deformation
	src1.create(Size(700,600), CV_8UC3);
    MLSDeformCurve(src1,a_p2d,stringrep);
//    testCompareContours(src1, a_p2d, stringrep);
}
