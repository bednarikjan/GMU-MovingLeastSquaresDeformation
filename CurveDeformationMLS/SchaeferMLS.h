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
    vector<vector<cv::Matx22d> > As;
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
    cl::Buffer* contourNewBuf_;
    cl::Buffer* keyPointsBuf_;
    cl::Buffer* shiftedkeyPointsBuf_;

    // debug
    cl::Buffer* debugBuf_;
    vector<double> debugVecA;
    vector<double> debugVecB;
	
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
            cv::Vec2d p_hat_i = P2V(p_hat[i]);
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

    /*!
     * \brief pointVec2vec Converts the cv::Point_<V> vector to type V vector.
     * \param pointVec          cv::Point_ vector.
     * \param serializedVec     type V vector
     */
    template<typename V>
    void pointVec2vec(std::vector<cv::Point_<V> >& pointVec, std::vector<T>& serializedVec) {
        if(serializedVec.size() != 2 * pointVec.size()) {
            std::cerr << "pointVec2vec: pointVec size must be 2 * serializedVec size." << std::endl;
            exit(1);
        }

        for(int i = 0; i < pointVec.size(); ++i) {
            serializedVec[(i << 1)]     = pointVec[i].x;
            serializedVec[(i << 1) + 1] = pointVec[i].y;
        }
    }

    /*!
     * \brief vec2PointVec Converts vector of type V to the vector of cv::Point_<V>
     * \param serializedVec input
     * \param pointVec output
     */
    template<typename V>
    void vec2PointVec(std::vector<V>& serializedVec, std::vector<cv::Point_<V> >& pointVec) {
        if(serializedVec.size() != 2 * pointVec.size()) {
            std::cerr << "vec2pointVec: pointVec size must be 2 * serializedVec size." << std::endl;
            exit(1);
        }

        for(int i = 0; i < pointVec.size(); ++i) {
            pointVec[i].x = serializedVec[(i << 1)];
            pointVec[i].y = serializedVec[(i << 1) + 1];
        }
    }

    // debug
    template<typename V>
    double comparePointVectors(std::vector<cv::Point_<V> >& a, std::vector<cv::Point_<V> >& b) {
        if(a.size() != b.size()) {
            std::cerr << "comparePointVectors: a and b vectors must be of the same size" << std::endl;
            return 1;
        }

        double sum = 0;
        for(int i = 0; i < a.size(); ++i) {
            V dx, dy;
            dx = a[i].x - b[i].x;
            dy = a[i].y - b[i].y;
            sum += std::sqrt(dx * dx + dy * dy);
        }

        return sum;
    }

    template<typename V>
    double compareVectors(std::vector<V>& a, std::vector<V>& b) {
        if(a.size() != b.size()) {
            std::cerr << "compareVectors: a and b vectors must be of the same size" << std::endl;
            return 1;
        }

        double sum = 0;
        for(int i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }

        return sum;
    }

public:
    const int MAX_CONTOUR_POINTS = 3000;
    const int MAX_KEY_POINTS = 250;

public:

    //! \brief SchaeferMLS Constructor
    //! \param useOpenCL Whether to use OpenCL or sequential code
    //!
    SchaeferMLS(bool useOpenCL = false) : usingOpenCL_(useOpenCL) {
        if(usingOpenCL_) {
            dc_ = new DeviceContext();
            if(initOpenCL() != ERR_SUCCESS) {
                std::cerr << "Failed to initializae OpenCL. Exiting now..." << std::endl;
                exit(1);
            }
        }
    }

    //! \brief SchaeferMLS Destructor
    //!
    ~SchaeferMLS() {
        if(dc_)                     delete dc_;
        if(kernel_)                 delete kernel_;
        if(contourBuf_)             delete contourBuf_;
        if(contourNewBuf_)          delete contourNewBuf_;
        if(keyPointsBuf_)           delete keyPointsBuf_;
        if(shiftedkeyPointsBuf_)    delete shiftedkeyPointsBuf_;
    }

    cl_int initOpenCL() {
        cl_int clErr = CL_SUCCESS;

        clErr = dc_->InitPlatform(CL_DEVICE_TYPE_GPU);
        if(clErr != ERR_SUCCESS)
        {
            std::cerr << "Failed to create OpenCL Context: " << WrapperErrorCodeToString(clErr) << std::endl;
            return clErr;
        }

        dc_->SetBaseKernelsPath("../../GMU-MovingLeastSquaresDeformation/CurveDeformationMLS/");

        std::cout << "Using: " << dc_->GetPlatform().getInfo<CL_PLATFORM_NAME>() << "\n" <<
                     dc_->GetDevice().getInfo<CL_DEVICE_NAME>() << std::endl;


        std::cout << "Loading kernels..." << std::endl;
        clErr = dc_->LoadProgram("deformShape.cl");
        if(clErr != ERR_SUCCESS)
        {
            std::cout << "Failed to load program: " << WrapperErrorCodeToString(clErr) << std::endl;
            return clErr;
        }

        kernel_ = new cl::Kernel(*(dc_->GetProgram("deformShape.cl")), "deformShape", &clErr);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to create Kernel: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_CREATION_FAILED;
        }

        contourBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_READ_ONLY, MAX_CONTOUR_POINTS * sizeof(cl_double2), NULL, &clErr);
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        contourNewBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_WRITE_ONLY, MAX_CONTOUR_POINTS * sizeof(cl_double2), NULL, &clErr);
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        keyPointsBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_READ_ONLY, MAX_KEY_POINTS * sizeof(cl_double2), NULL, &clErr);
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        shiftedkeyPointsBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_READ_ONLY, MAX_KEY_POINTS * sizeof(cl_double2), NULL, &clErr);
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        debugBuf_ = new cl::Buffer(dc_->GetContext(), CL_MEM_WRITE_ONLY, MAX_CONTOUR_POINTS * MAX_KEY_POINTS * sizeof(cl_double), NULL, &clErr);
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
		
//		A.create(m_curve.size(),control_idx.size());
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

        // debug
        debugVecA.clear();

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

        // debug
        // p_star
//        debugVecA.resize(m_curve.size() * 2);

        // p_hat
//        debugVecA.resize(m_curve.size() * control_idx.size() * 2);

        // mu_s
//        debugVecA.resize(m_curve.size());

        for (int i=0; i<m_curve.size(); i++) {
            cv::Vec2d v = P2V(m_curve[i]);

            cv::Point2d p_star = GetWeightedMeanForPoint(i,p);
            std::vector<cv::Point2d> p_hat = GetCurveAroundMean(p,p_star);
            cv::Point2d q_star = GetWeightedMeanForPoint(i,q);
            std::vector<cv::Point2d> q_hat = GetCurveAroundMean(q,q_star);

            // debug
            // p_star
//            debugVecA[2 * i] = p_star.x;
//            debugVecA[2 * i + 1] = p_star.y;

            // p_hat
//            std::vector<double> tmp;
//            tmp.resize(control_idx.size() * 2);
//            pointVec2vec(p_hat, tmp);
//            for(int j = 0; j < control_idx.size(); ++j) {
//                debugVecA[i * control_idx.size() * 2 + j * 2] = tmp[j * 2];
//                debugVecA[i * control_idx.size() * 2 + j * 2 + 1] = tmp[j * 2 + 1];
//            }

            cv::Point2d newpoint(0,0);

            //Similarity - section 2.2
            double mu_s = GetWeightedCovarSum(p_hat, m_w[i]);

            // debug - mu_s
//            debugVecA[i] = mu_s;

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

    cl_int deformCurveOneStepParallel(const vector<Point_<T> >& curve, const vector<int>& control_idx, const std::vector<cv::Point_<T> >& shifts) {
        cl_int clErr;


        // Set problem size
        cl::NDRange lWorkItems(64);
        cl::NDRange gWorkItems(DeviceContext::iCeilTo(m_curve.size(), lWorkItems[0]));

        // debug
        std::cout << "workgoup size: " << lWorkItems[0] << endl;
        std::cout << "work items   : " << gWorkItems[0] << endl;

        cl::Event copyContourPoints;
        cl::Event copyContourPointsNew;
        cl::Event copyKeyPoints;
        cl::Event copyKeyPointsNew;
        cl::Event deformShapeKernelRun;

        // Prepare global memory buffer used by the kernel to storu temp computation results
        std::vector<cl::Buffer> buffers;
        buffers.push_back(cl::Buffer(dc_->GetContext(), CL_MEM_READ_WRITE, control_idx.size() * m_curve.size() * sizeof(cl_double), NULL, &clErr));
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        buffers.push_back(cl::Buffer(dc_->GetContext(), CL_MEM_READ_WRITE, control_idx.size() * m_curve.size() * sizeof(cl_double2), NULL, &clErr));
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        buffers.push_back(cl::Buffer(dc_->GetContext(), CL_MEM_READ_WRITE, control_idx.size() * m_curve.size() * sizeof(cl_double2), NULL, &clErr));
        if(clErr)
        {
            std::cout << "Failed to create Buffer: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_CREATION_FAILED;
        }

        // debug
        m_original_curve = m_curve;
        debugVecB.clear();

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

        if(m_curve.size() > MAX_CONTOUR_POINTS) {
            std::cerr << "Contour can not contain more than " << MAX_CONTOUR_POINTS << " points." << endl;
            return ERR_GENERIC_ERROR;
        }

        if(control_pts.size() > MAX_KEY_POINTS) {
            std::cerr << "Key points can not contain more than " << MAX_KEY_POINTS << " points." << endl;
        }

        // Convert cv::Point<T> vector to T vector
        std::vector<double> contourPoints;
        contourPoints.resize(m_curve.size() * 2);
        pointVec2vec<double>(m_curve, contourPoints);

        // Copy contour points to device
        clErr = dc_->GetCmdQueue().enqueueWriteBuffer(*contourBuf_, CL_TRUE, 0, ((int)contourPoints.size() >> 1) * sizeof(cl_double2), &(contourPoints[0]), NULL, &copyContourPoints);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer write failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }        

        // Copy key points to device
        std::vector<double> keyPoints;
        keyPoints.resize(control_pts.size() * 2);
        pointVec2vec<double>(control_pts, keyPoints);

        clErr = dc_->GetCmdQueue().enqueueWriteBuffer(*keyPointsBuf_, CL_TRUE, 0, ((int)keyPoints.size() >> 1) * sizeof(cl_double2), &(keyPoints[0]), NULL, &copyKeyPoints);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer write failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }

        // Copy new key points to device
        std::vector<double> keyPointsNew;
        keyPointsNew.resize(deformed_control_pts.size() * 2);
        pointVec2vec<double>(deformed_control_pts, keyPointsNew);

        clErr = dc_->GetCmdQueue().enqueueWriteBuffer(*shiftedkeyPointsBuf_, CL_TRUE, 0, ((int)keyPointsNew.size() >> 1) * sizeof(cl_double2), &(keyPointsNew[0]), NULL, &copyKeyPointsNew);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer write failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }

        // Set kernel arguments
        // contour points
        clErr = kernel_->setArg(0, *contourBuf_);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 0: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // number of contour points
        clErr = kernel_->setArg(1, (cl_int)((int)contourPoints.size() >> 1));
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 1: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // key points
        clErr = kernel_->setArg(2, *keyPointsBuf_);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 2: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // new key points
        clErr = kernel_->setArg(3, *shiftedkeyPointsBuf_);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 3: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // key points size
        clErr = kernel_->setArg(4, (cl_int)((int)keyPoints.size() >> 1));
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 4: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // key points size
        clErr = kernel_->setArg(5, *contourNewBuf_);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 5: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

//        clErr = kernel_->setArg(6, cl::Local(sizeof(cl_double) * lWorkItems[0] * control_pts.size()));
//        if(clErr != CL_SUCCESS)
//        {
//            std::cout << "Failed to set kernel argument 6: " << OpenCLErrorCodeToString(clErr) << std::endl;
//            return ERR_KERNEL_ARGSET_FAILED;
//        }

        // m_w
        clErr = kernel_->setArg(6, buffers[0]);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 6: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // p_hat
        clErr = kernel_->setArg(7, buffers[1]);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 7: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // q_hat
        clErr = kernel_->setArg(8, buffers[2]);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 8: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }

        // debug buffer
        clErr = kernel_->setArg(9, *debugBuf_);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to set kernel argument 9: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_KERNEL_ARGSET_FAILED;
        }


        // Run kernel
        clErr = dc_->GetCmdQueue().enqueueNDRangeKernel(*kernel_, cl::NullRange, gWorkItems, lWorkItems, NULL, &deformShapeKernelRun);
        if(clErr != CL_SUCCESS)
        {
            std::cout << "Failed to run kernel: " << OpenCLErrorCodeToString(clErr);
            return ERR_KERNEL_RUN_FAILED;
        }       

        deformShapeKernelRun.wait();
        std::cout << "Time elapsed (device): " << (double)(deformShapeKernelRun.getProfilingInfo<CL_PROFILING_COMMAND_END>() - deformShapeKernelRun.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0 << " ms\n";

        // Copy new contour points from device
        std::vector<double> contourPointsNew;
        contourPointsNew.resize(contourPoints.size());

        clErr = dc_->GetCmdQueue().enqueueReadBuffer(*contourNewBuf_, CL_TRUE, 0, ((int)contourPointsNew.size() >> 1) * sizeof(cl_double2), &(contourPointsNew[0]), NULL, &copyContourPointsNew);

        if(clErr != CL_SUCCESS)
        {
            std::cout << "Buffer read failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
            return ERR_BUFFER_WRITE_FAILED;
        }
//        vec2PointVec(contourPointsNew, m_curve);

        // debug - compare weight
//        debugVecB.resize(m_curve.size() * control_idx.size());
//        clErr = dc_->GetCmdQueue().enqueueReadBuffer(*debugBuf_, CL_TRUE, 0, m_curve.size() * control_idx.size() * sizeof(cl_double), &(debugVecB[0]), NULL, NULL);
//        if(clErr != CL_SUCCESS)
//        {
//            std::cout << "Buffer read failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
//            return ERR_BUFFER_WRITE_FAILED;
//        }
//        double sum = 0.0;
//        for(int i = 0; i < m_w.size(); ++i) {
//            for(int j = 0; j < m_w[0].size(); ++j) {
//                sum += std::abs(m_w[i][j] - debugVecB[i * m_w[0].size() + j]);
//            }
//        }
//        std::cout << "Compare m_w and m_w_device: " << sum << std::endl;


        // debug - compare p_star
//        debugVecB.resize(m_curve.size() * 2);
//        clErr = dc_->GetCmdQueue().enqueueReadBuffer(*debugBuf_, CL_TRUE, 0, m_curve.size() * 2 * sizeof(cl_double), &(debugVecB[0]), NULL, NULL);
//        if(clErr != CL_SUCCESS)
//        {
//            std::cout << "Buffer read failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
//            return ERR_BUFFER_WRITE_FAILED;
//        }
//        cout << "Compare getWeightedMeanForPoint: " <<
//                compareVectors(debugVecA, debugVecB) <<
//                std::endl;

        // debug - compare p_hat
//        debugVecB.resize(m_curve.size() * control_idx.size() * 2);
//        clErr = dc_->GetCmdQueue().enqueueReadBuffer(*debugBuf_, CL_TRUE, 0, debugVecB.size() * sizeof(cl_double), &(debugVecB[0]), NULL, NULL);
//        if(clErr != CL_SUCCESS)
//        {
//            std::cout << "Buffer read failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
//            return ERR_BUFFER_WRITE_FAILED;
//        }
//        cout << "Compare getCurveAroundMean: " <<
//                compareVectors(debugVecA, debugVecB) <<
//                std::endl;

        // debug - compare mu_s
//        debugVecB.resize(m_curve.size());
//        clErr = dc_->GetCmdQueue().enqueueReadBuffer(*debugBuf_, CL_TRUE, 0, debugVecB.size() * sizeof(cl_double), &(debugVecB[0]), NULL, NULL);
//        if(clErr != CL_SUCCESS)
//        {
//            std::cout << "Buffer read failed: " << OpenCLErrorCodeToString(clErr) << std::endl;
//            return ERR_BUFFER_WRITE_FAILED;
//        }
//        cout << "Compare mu_s: " <<
//                compareVectors(debugVecA, debugVecB) <<
//                std::endl;

        // debug - print the time of copying the data to/from the device
        std::cout << "Time elapsed (copy to/from device): " <<
                     (DeviceContext::getEventTime(copyContourPoints) +
                      DeviceContext::getEventTime(copyKeyPoints) +
                      DeviceContext::getEventTime(copyKeyPointsNew)) +
                      DeviceContext::getEventTime(copyContourPointsNew) <<
                     " ms\n";

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
