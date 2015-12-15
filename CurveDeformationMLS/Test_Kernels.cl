__kernel void TestThroughtput(__global float2 *pPoints,
    __constant float2 *pKeyPoints, int keyPointsCount)
{
    int idx = get_global_id(0);

    float2 center = (float2)(0.0f);

    for(int i = 0; i < keyPointsCount; ++i)
    {
        center += pKeyPoints[i];
    }

    pPoints[idx] = center / (float)keyPointsCount;
}
