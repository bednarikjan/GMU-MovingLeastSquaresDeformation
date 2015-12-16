__kernel void deformShape(__global double2 *contourPoints, int contourPointsCount)
{

    int idx = get_global_id(0);

    if(idx < contourPointsCount) {
        double2 center = (double2)(0.0f);

        for(int i = 0; i < contourPointsCount; ++i) {
            center += contourPoints[i];
        }

        center = center / (double)contourPointsCount;
    }
}
