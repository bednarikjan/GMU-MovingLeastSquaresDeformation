/*
 *  SchaeferMLS.h
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 12/28/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 *	Implementing "Image deformation using moving least squares", S. Schaefer et al. 2006
 *  http://dl.acm.org/citation.cfm?id=1141920
 *	(for 2D curves)
 *
 */

#include <numeric>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include "CurveCSS.h"

// Our OpenCL wrapper
#include "device_context.h"
#include "error_codes.h"

#define P2V(p) Vec2d((p).x,(p).y)
#define V2P(v) Point2d((v)[0],(v)[1])

template<typename T>
class SchaeferMLS {
private:
    Mat_<double> A;
	vector<vector<Matx22d> > As;
	vector<double> mu_s;
	vector<double> mu_r;
	vector<Point_<T> > control_pts;
	vector<Point_<T> > deformed_control_pts;
	vector<Point_<T> > m_curve;
	vector<Point_<T> > m_original_curve;
//	double m_original_curve_scale;
	vector<vector<double> > m_w;
	vector<Vec2d> m_vmp;

    // OpenCL
    bool usingOpenCL_;
    DeviceContext* dc_;
    cl::Kernel* kernel_;
    cl::Buffer* contourBuf_;
    cl::Buffer* keyPointsBuf_;
    cl::Buffer* shiftedkeyPointsBuf_;
	
	template<typename V>
	vector<Point2d> GetCurveAroundMean(const vector<Point_<V> >& p, const Point_<V>& p_star) {
		vector<Point2d> p_hat(p.size());
		for (int i=0; i<p.size(); i++) p_hat[i] = p[i] - p_star;
		return p_hat;
	}
	
	void CalcWeights() {
		m_w.resize(m_curve.size());
		for (int v=0; v<m_curve.size(); v++) {
			m_w[v].resize(control_pts.size());
			for (int i=0; i<control_pts.size(); i++) {
				double d = pow(norm(control_pts[i] - m_curve[v]),3.0);
				if (d < 1.0) {
					m_w[v][i] = 1.0;
				} else {
					m_w[v][i] = 1.0 / d;
				}
			}
		}
	}
	void CalcGeodesicWeights(const vector<int>& ctrl_pts_curve_idx) {
		m_w.resize(m_curve.size());
		for (int v=0; v<m_curve.size(); v++) {
			m_w[v].resize(control_pts.size());
			for (int i=0; i<ctrl_pts_curve_idx.size(); i++) {
				double d = pow(abs(ctrl_pts_curve_idx[i] - v),2.0);
				if (d < 1.0) {
					m_w[v][i] = 1.0;
				} else {
					m_w[v][i] = 1.0 / d;
				}
			}
		}
	}
	
	template<typename Q>
	Point2d GetWeightedMeanForPoint(int v, const vector<Point_<Q> >& q) {
		Point2d q_star(0,0);
		double sum_w_i = 0.0;
		for (int i=0; i<q.size(); i++) {
			q_star += q[i] * m_w[v][i];
			sum_w_i += m_w[v][i];
		}
		
		return q_star * (1.0/sum_w_i);
	}
	
	double GetWeightedCovarSum(const vector<Point2d>& p_hat, const vector<double>& w) {
		double mean = 0;
		for (int i=0; i<p_hat.size(); i++) {
			Vec2d p_hat_i = P2V(p_hat[i]);
			double prod = (p_hat_i.t() * p_hat_i).val[0];
			mean += (prod * w[i]);
		}
		return mean;
	}
	
	template<typename V>
	Matx22d GetWeightedCovarMat(const vector<Point_<V> >& p_hat, const vector<double>& w) {
		Matx22d covarmat(0.0);
		for (int i=0; i<p_hat.size(); i++) {
			Vec2d p_hat_i = P2V(p_hat[i]);
			covarmat = covarmat + p_hat_i * p_hat_i.t() * w[i];
		}
		covarmat = covarmat.inv();
		return covarmat;
	}
public:
    SchaeferMLS(bool useOpenCL = false) : usingOpenCL_(useOpenCL) {
        if(usingOpenCL_) {
            dc_ = new DeviceContext();
            if(initOpenCL() != ERR_SUCCESS) {
                std::cerr << "Failed to initializae OpenCL. Exiting now..." << std::endl;
                exit(1);
            }
        }
    }

    cl_int initOpenCL() {
        cl_int clErr = CL_SUCCESS;

        clErr = dc_->InitPlatform(CL_DEVICE_TYPE_GPU);
        if(clErr != ERR_SUCCESS)
        {
            std::cerr << "Failed to create OpenCL Context: " << WrapperErrorCodeToString(clErr) << std::endl;
            return clErr;
        }

        dc_->SetBaseKernelsPath("../GMU-MovingLeastSquaresDeformation/CurveDeformationMLS/");

        std::cout << "Using: " << dc_->GetPlatform().getInfo<CL_PLATFORM_NAME>() << "\n" <<
                     dc_->GetDevice().getInfo<CL_DEVICE_NAME>() << std::endl;


        std::cout << "Loading kernels..." << std::endl;
        clErr = dc_->LoadProgram("Test_Kernels.cl");
        if(clErr != ERR_SUCCESS)
        {
            std::cout << "Failed to load program: " << WrapperErrorCodeToString(clErr) << std::endl;
            return clErr;
        }

        kernel_ = new cl::Kernel(*(dc_->GetProgram("Test_Kernels.cl")), "TestThroughtput", &clErr);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to create Kernel: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_CREATION_FAILED;
        }

        contourBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_READ_ONLY, 1000 * sizeof(cl_double2), NULL, &clErr);
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }
    }

    void Init(const vector<Point_<T> >& curve, const vector<int>& control_idx) {
		As.clear();
		mu_s.clear();
		mu_r.clear();
		control_pts.clear();
		deformed_control_pts.clear();
		m_curve.clear();
		m_original_curve.clear();
		m_w.clear();
		m_vmp.clear();
		
		m_curve = curve;
		m_original_curve = curve;
		
//		Mat a_ = Mat(m_original_curve).reshape(1);
//		PCA a_pca(a_,Mat(),CV_PCA_DATA_AS_ROW);
//		m_original_curve_scale = sqrt(a_pca.eigenvalues.at<double>(0));
		
		vector<Point2d> p;
		for (int i=0; i<control_idx.size(); i++) {
			p.push_back(m_curve[control_idx[i]]);
			control_pts.push_back(m_curve[control_idx[i]]);
		}
		deformed_control_pts = control_pts;
		
		CalcWeights();
//		CalcGeodesicWeights(control_idx);
		
		A.create(m_curve.size(),control_idx.size());
		As.resize(m_curve.size(),vector<Matx22d>(control_idx.size()));
		m_vmp.resize(m_curve.size());
		
		for (int i=0; i<m_curve.size(); i++) {
			Vec2d v = P2V(m_curve[i]);
			
			Point2d p_star = GetWeightedMeanForPoint(i,p);
			vector<Point2d> p_hat = GetCurveAroundMean(p,p_star);
			
			//Affine - section 2.1
//			for (int j=0; j<p.size(); j++) {
//				Matx22d covmatx = GetWeightedCovarMat(p_hat,m_w[i]);
//				Matx12d vmp = (v - P2V(p_star)).t();
//				Matx21d p_hat_j_t = P2V(p_hat[j]);
//				A(i,j) = (vmp * covmatx * p_hat_j_t).val[0] * m_w[i][j];
//			}
			
			//Similarity - section 2.2
			double mu_s = GetWeightedCovarSum(p_hat, m_w[i]);
			for (int j=0; j<p.size(); j++) {
				//eqn (7)
				Matx22d lh(p_hat[j].x,p_hat[j].y,p_hat[j].y,-p_hat[j].x);
				m_vmp[i] = (v - P2V(p_star));
				Matx22d rh(m_vmp[i][0],m_vmp[i][1],m_vmp[i][1],-m_vmp[i][0]);
				
				As[i][j] = lh * rh.t() * (m_w[i][j] / mu_s);
			}
		}
	}
	
	void UpdateAffine() {
		vector<Point2d> q; ConvertCurve(deformed_control_pts, q);
		
		for (int i=0; i<m_curve.size(); i++) {
			Point2d q_star = GetWeightedMeanForPoint(i,q);
			vector<Point2d> q_hat = GetCurveAroundMean(q,q_star);

//			cout << A.row(i) << endl;
			Point2d newpoint(0,0);
			for (int j=0; j<q.size(); j++) {
				newpoint += q_hat[j] * A(i,j);
			}
			newpoint += q_star;
			
			m_curve[i] = Point_<T>(newpoint.x,newpoint.y);
		}
	}
	
	void UpdateSimilarity() {
		vector<Point2d> q; ConvertCurve(deformed_control_pts, q);
		
		for (int i=0; i<m_curve.size(); i++) {
			vector<double> w;
			Point2d q_star = GetWeightedMeanForPoint(i,q);
			vector<Point2d> q_hat = GetCurveAroundMean(q,q_star);
			
			Point2d newpoint(0,0);
			for (int j=0; j<q.size(); j++) {
				Matx22d as_i_j = As[i][j];
				Matx12d q_hat_j(q_hat[j].x,q_hat[j].y);
				Matx12d prod = q_hat_j * as_i_j;
				newpoint += Point2d(prod.val[0],prod.val[1]);
			}
			newpoint += q_star;
			
			m_curve[i] = Point_<T>(newpoint.x,newpoint.y);
		}
	}
	
	void UpdateRigid() {
		vector<Point2d> q; ConvertCurve(deformed_control_pts, q);
		
		for (int i=0; i<m_curve.size(); i++) {
			vector<double> w;
			Point2d q_star = GetWeightedMeanForPoint(i,q);
			vector<Point2d> q_hat = GetCurveAroundMean(q,q_star);
			
			Point2d newpoint(0,0);
			
			//calc sum_i(q_hat[i] * A[i]) 
			for (int j=0; j<q.size(); j++) {
				Matx22d as_i_j = As[i][j];
				Matx12d q_hat_j(q_hat[j].x,q_hat[j].y);
				Matx12d prod = q_hat_j * as_i_j;
				newpoint += Point2d(prod.val[0],prod.val[1]);
			}
			
			//eqn (8)
			double scale = norm(m_vmp[i]) / norm(newpoint);
			newpoint = newpoint * scale + q_star;
			
			m_curve[i] = Point_<T>(newpoint.x,newpoint.y);
		}		
	}

    void deformCurveOneStep(const vector<Point_<T> >& curve, const vector<int>& control_idx, const std::vector<cv::Point_<T> >& shifts) {
        As.clear();
//        mu_s.clear();
//        mu_r.clear();
        control_pts.clear();
        deformed_control_pts.clear();
        m_curve.clear();
        m_original_curve.clear();
        m_w.clear();
        m_vmp.clear();

        m_curve = curve;
        m_original_curve = curve;

        control_pts.reserve(control_idx.size());
        deformed_control_pts.reserve(control_idx.size());
        for (int i=0; i<control_idx.size(); i++) {
            control_pts.push_back(m_curve[control_idx[i]]);
            deformed_control_pts.push_back(m_curve[control_idx[i]] + shifts[i]);
        }
        const std::vector<cv::Point2d> &p = control_pts;

        const std::vector<cv::Point2d> &q = deformed_control_pts; // ConvertCurve(deformed_control_pts, q)

        CalcWeights();

        As.resize(m_curve.size(),std::vector<cv::Matx22d>(control_idx.size()));
        m_vmp.resize(m_curve.size());

        for (int i=0; i<m_curve.size(); i++) {
            cv::Vec2d v = P2V(m_curve[i]);

            cv::Point2d p_star = GetWeightedMeanForPoint(i,p);
            std::vector<cv::Point2d> p_hat = GetCurveAroundMean(p,p_star);
            cv::Point2d q_star = GetWeightedMeanForPoint(i,q);
            std::vector<cv::Point2d> q_hat = GetCurveAroundMean(q,q_star);

            cv::Point2d newpoint(0,0);

            //Similarity - section 2.2
            double mu_s = GetWeightedCovarSum(p_hat, m_w[i]);
            for (int j=0; j<p.size(); j++) {
                //eqn (7)
                cv::Matx22d lh(p_hat[j].x,p_hat[j].y,p_hat[j].y,-p_hat[j].x);
                m_vmp[i] = (v - P2V(p_star));
                cv::Matx22d rh(m_vmp[i][0],m_vmp[i][1],m_vmp[i][1],-m_vmp[i][0]);

                As[i][j] = lh * rh.t() * (m_w[i][j] / mu_s);

                cv::Matx22d as_i_j = As[i][j];
                cv::Matx12d q_hat_j(q_hat[j].x,q_hat[j].y);
                cv::Matx12d prod = q_hat_j * as_i_j;
                newpoint += cv::Point2d(prod.val[0],prod.val[1]);
            }


            double scale = norm(m_vmp[i]) / norm(newpoint);
            newpoint = newpoint * scale + q_star;

            m_curve[i] = cv::Point_<T>(newpoint.x,newpoint.y); // setting the result
        }
    }

    void deformCurveOneStepParallel(const vector<Point_<T> >& curve, const vector<int>& control_idx, const std::vector<cv::Point_<T> >& shifts) {
        cl_int clErr;

        control_pts.clear();
        deformed_control_pts.clear();
        m_curve.clear();
        m_curve = curve;

        control_pts.reserve(control_idx.size());
        deformed_control_pts.reserve(control_idx.size());
        for (int i=0; i<control_idx.size(); i++) {
            control_pts.push_back(m_curve[control_idx[i]]);
            deformed_control_pts.push_back(m_curve[control_idx[i]] + shifts[i]);
        }

        cl::Event profCopyEvent;
        clErr = dc_->GetCmdQueue().enqueueWriteBuffer(contourBuf_, CL_TRUE, 0, 1000 * sizeof(cl_double2), &keyPoints[0], NULL, &profCopyEvent);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer write failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }

        std::cout << "Copy elapsed: " << (double)(profCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0 << " ms\n";

        clErr = deviceContext.GetCmdQueue().enqueueWriteBuffer(conturePointsBuffer, CL_TRUE, 0, 512 * sizeof(cl_float2), &conturePoints[0], NULL, &profCopyEvent);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer write failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }

        std::cout << "Copy elapsed: " << (double)(profCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0 << " ms\n";

        clErr = testKernel.setArg(0, conturePointsBuffer);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        clErr = testKernel.setArg(1, keyPointsBuffer);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        clErr = testKernel.setArg(2, 32);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        cl::NDRange gWorkItems(512);
        cl::NDRange lWorkItems = cl::NullRange;

        cl::Event profEvent;
        clErr = deviceContext.GetCmdQueue().enqueueNDRangeKernel(testKernel, cl::NullRange, gWorkItems, lWorkItems, NULL, &profEvent);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to run kernel: " << OpenCLErrorCodeToString(clErr);
            return ERR_KERNEL_RUN_FAILED;
        }

        profEvent.wait();
        std::cout << "Elaped: " << (double)(profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0 << " ms\n";

    }
	
	const vector<Point_<T> >& GetControlPts() { return control_pts; }
	vector<Point_<T> >& GetDeformedControlPts() { return deformed_control_pts; }
    vector<Point_<T> > GetContourPts() { return m_curve; }
	
	void Draw(Mat& img) {
//		img.setTo(0);
//		{
//			//draw small original
//			vector<Point2d> tmp_curve;
//			cv::transform(m_original_curve,tmp_curve,getRotationMatrix2D(Point2f(0,0),0,50/m_original_curve_scale));
////			Mat tmp_curve_m(tmp_curve); tmp_curve_m += Scalar(25,0);
			
//			drawOpenCurve(img, tmp_curve, Scalar::all(255), 2);
//		}
		drawOpenCurve(img, m_curve, Scalar(0,0,255), 2);
		for (int i=0; i<control_pts.size(); i++) {
//			circle(img, control_pts[i], 3, Scalar(0,0,255), 1);
			circle(img, deformed_control_pts[i], 5, Scalar(0,255,255), 2);
		}
	}
};
