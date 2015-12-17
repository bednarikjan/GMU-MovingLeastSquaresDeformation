//======== Copyright (c) 2008-2015, Filip Vaverka, All rights reserved. ======//
//
// Purpose:     OpenCL device and context wrapper
//
// $NoKeywords: $OpenCLStub $device_context.cpp
// $Date:       $2015-11-18
//============================================================================//

#include "device_context.h"
#include "error_codes.h"

#include <vector>
#include <map>
#include <string>
#include <fstream>

#include "CL/cl.hpp"

using namespace std;



DeviceContext::DeviceContext()
    : m_clError(CL_SUCCESS), m_clCommandQueueProps(0)
{
    m_clCommandQueueProps |= CL_QUEUE_PROFILING_ENABLE;
}

DeviceContext::~DeviceContext()
{

}

int DeviceContext::InitPlatform(cl_device_type deviceType, FuncPlatformSelect_t pfnPlatformSelect,
                                FuncDeviceSelect_t pfnDeviceSelect)
{
    m_clError = cl::Platform::get(&m_platforms);
    if(m_clError != CL_SUCCESS || m_platforms.size() < 1)
        return ERR_GET_PLATFORMS_FAILED;

    size_t platformId = 0;
    if(pfnPlatformSelect)
        platformId = pfnPlatformSelect(m_platforms);

    m_clPlatform = m_platforms[platformId];

    m_ctxProps[0] = CL_CONTEXT_PLATFORM;
    m_ctxProps[1] = (cl_context_properties)(m_platforms[platformId])();
    m_ctxProps[2] = 0;

    m_clContext = cl::Context(deviceType, m_ctxProps, NULL, NULL, &m_clError);
    if(m_clError != CL_SUCCESS)
        return ERR_OCL_CONTEXT_CREATE_FAILED;

    m_clDevices = m_clContext.getInfo<CL_CONTEXT_DEVICES>(&m_clError);
    if(m_clError != CL_SUCCESS || m_clDevices.size() < 1)
        return ERR_OCL_GET_DEVICES_FAILED;

    size_t deviceId = 0;
    if(pfnDeviceSelect)
        deviceId = pfnDeviceSelect(m_clDevices);

    m_clDevice = m_clDevices[deviceId];

    m_clDefaultCommandQueue = cl::CommandQueue(m_clContext, m_clDevice,
                                               m_clCommandQueueProps, &m_clError);
    if(m_clError != CL_SUCCESS)
        return ERR_OCL_CMDQUEUE_INIT_FAILED;

    return ERR_SUCCESS;
}

int DeviceContext::LoadProgram(const char *pszFilename)
{
    if(m_programs.find(string(pszFilename)) != m_programs.end())
        return ERR_SUCCESS;

    fstream file((m_baseKernelsPath + pszFilename).c_str(),
                 fstream::in | fstream::binary);
    if(!file)
        return ERR_PROGRAM_FILE_OPEN_FAILED;

    size_t nFileLen = 0;

    file.seekg(0, fstream::end);
    nFileLen = (size_t)file.tellg();
    file.seekg(0);

    string strKernelSource(nFileLen, ' ');
    file.read(&strKernelSource[0], nFileLen);
    file.close();

    cl::Program::Sources sources(1, std::make_pair(strKernelSource.c_str(), nFileLen));
    cl::Program program(m_clContext, sources, &m_clError);

    if(m_clError != CL_SUCCESS)
        return ERR_OCL_PROGRAM_CREATION_FAILED;

    m_clError = program.build(m_clDevices, NULL, NULL, NULL);

    if(m_clError != CL_SUCCESS)
    {
        if(m_clError == CL_BUILD_PROGRAM_FAILURE)
            m_clBuildErrorInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_clDevice);

        return ERR_OCL_PROGRAM_BUILD_FAILED;
    }

    m_programs.insert(ProgramsMapPair_t(string(pszFilename), program));

    return ERR_SUCCESS;
}

double DeviceContext::getEventTime(cl::Event i_event)
{
    return double(i_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                  i_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0;
}

unsigned int DeviceContext::iCeilTo(unsigned int data, unsigned int align_size)
{
    return ((data + align_size - 1) / align_size) * align_size;
}
