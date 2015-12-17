double2 getWeightedMeanForPoint(    int idxG,
                                    __constant double2 *keyPoints,
                                    int numKeyPoints,
                                    __global double *m_w)
{
    double2 q_star = (double2)(0.0, 0.0);
    double sum_w = 0.0;
    for(int i = 0; i < numKeyPoints; ++i) {
        double weight = m_w[idxG * numKeyPoints + i];
        q_star += keyPoints[i] * weight;
        sum_w += weight;
    }
    q_star /= sum_w;

    return q_star;
}

void getCurveAroundMean(int idxG,
                        __constant double2 *keyPoints,
                        int numKeyPoints,
                        double2 q_star,
                        __global double2 *q_hat)
{
    for(int i = 0; i < numKeyPoints; ++i) {
        q_hat[idxG * numKeyPoints + i] = keyPoints[i] - q_star;
    }
}


__kernel void deformShape(  __constant double2 *contourPoints,
                            int numContourPoints,
                            __constant double2 *keyPoints,
                            __constant double2 *keyPointsNew,
                            int numKeyPoints,
                            __global double2 *contourPointsNew,
                            __global double *m_w,
                            __global double2 *p_hat,
                            __global double2 *q_hat,
                            __global double *debugData)
{
    // Work item indices.
    int idxG = get_global_id(0);
    int idxL = get_local_id(0);



    if(idxG < numContourPoints) {
        // Compute weights.
        for(int i = 0; i < numKeyPoints; ++i) {
            double d = pow(distance(keyPoints[i], contourPoints[idxG]), (double)3.0);
            double weight = (d < 1.0) ? 1.0 : 1.0 / d;
            m_w[idxG * numKeyPoints + i] = weight;

            // debug
//            debugData[idxG * numKeyPoints + i] = weight;
        }

        // Compute weighted mean for point (p_star) and curve around mean (p_hat)
        double2 p_star = getWeightedMeanForPoint(idxG, keyPoints, numKeyPoints, m_w);
        getCurveAroundMean(idxG, keyPoints, numKeyPoints, p_star, p_hat);

        // Compute weighted mean for point (q_star) and curve around mean (q_hat)
        double2 q_star = getWeightedMeanForPoint(idxG, keyPointsNew, numKeyPoints, m_w);
        getCurveAroundMean(idxG, keyPointsNew, numKeyPoints, q_star, q_hat);

        // debug - p_star
//        debugData[2 * idxG] = p_star.x;
//        debugData[2 * idxG + 1] = p_star.y;

        // debug - p_hat
//        for(int i = 0; i < numKeyPoints; ++i) {
//            debugData[idxG * numKeyPoints * 2 + i * 2] = p_hat[idxG * numKeyPoints + i].x;
//            debugData[idxG * numKeyPoints * 2 + i * 2 + 1] = p_hat[idxG * numKeyPoints + i].y;
//        }

        // Get weighted covariance sum
        double mu_s = 0.0;
        for(int i = 0; i < numKeyPoints; ++i) {
             double2 p_hat_i = p_hat[idxG * numKeyPoints + i];
             double prod = p_hat_i.x * p_hat_i.x + p_hat_i.y * p_hat_i.y;
             mu_s += prod * m_w[idxG * numKeyPoints + i];
        }

        // debug mu_s
//        debugData[idxG] = mu_s;

        double2 newpoint    = (double2)(0.0);
        double2 m_vmp       = contourPoints[idxG] - p_star;

        for(int i = 0; i < numKeyPoints; ++i) {
            double2 p_hat_i = p_hat[idxG * numKeyPoints + i];
            double4 lh = (double4)(p_hat_i.x, p_hat_i.y, p_hat_i.y, -p_hat_i.x);
            double4 rh = (double4)(m_vmp.x, m_vmp.y, m_vmp.y, -m_vmp.x);

            double m_wDmu_s = m_w[idxG * numKeyPoints + i] / mu_s;
            double4 As = (double4)( (lh.x * rh.x + lh.y * rh.y) * m_wDmu_s,
                                    (lh.x * rh.z + lh.y * rh.w) * m_wDmu_s,
                                    (lh.z * rh.x + lh.w * rh.y) * m_wDmu_s,
                                    (lh.z * rh.z + lh.w * rh.w) * m_wDmu_s);
        }

    }
}


