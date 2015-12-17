//======== Copyright (c) 2008-2015, Filip Vaverka, All rights reserved. ======//
//
// Purpose:     OpenCL device and context wrapper
//
// $NoKeywords: $OpenCLStub $device_context.h
// $Date:       $2015-11-18
//============================================================================//

#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

// System headers
#include <string>
#include <map>
#include <vector>

// OpenCL headers
#include "CL/cl.hpp"

class DeviceContext
{
public:
    typedef size_t (*FuncPlatformSelect_t)(const std::vector<cl::Platform> &platforms);
    typedef size_t (*FuncDeviceSelect_t)(const std::vector<cl::Device> &devices);

    DeviceContext();
    ~DeviceContext();

    int InitPlatform(cl_device_type deviceType, FuncPlatformSelect_t pfnPlatformSelect = NULL,
                     FuncDeviceSelect_t pfnDeviceSelect = NULL);

    int LoadProgram(const char *pszFilename);

    cl::Device &GetDevice() { return m_clDevice; }
    cl::Context &GetContext() { return m_clContext; }
    cl::Platform &GetPlatform() { return m_clPlatform; }
    cl::CommandQueue &GetCmdQueue() { return m_clDefaultCommandQueue; }

    cl::Program *GetProgram(const std::string &name)
    {
        ProgramsMap_t::iterator i = m_programs.find(name);
        if(i == m_programs.end())
            return NULL;

        return &i->second;
    }

    void SetBaseKernelsPath(const char *pszBasePath) { m_baseKernelsPath = pszBasePath; }
    const char *GetBaseKernelsPath() const { return m_baseKernelsPath.c_str(); }

    cl_int GetLastOpenCLError() const { return m_clError; }

    const std::string &GetBuildErrorInfo() const { return m_clBuildErrorInfo; }

    //! \brief getEventTime Returns time span of the given 'i_event' (start to end) in [ms].
    //! \param i_event
    //! \return
    //!
    static double getEventTime(cl::Event i_event);

    //! \brief iCeilTo Aligns work items count (int the given dimension) to the
    //! integral multiply of the workgroup size (in the given dimension)
    //! \param data
    //! \param align_size
    //! \return
    //!
    static unsigned int iCeilTo(unsigned int data, unsigned int align_size);

protected:
    typedef std::map<std::string, cl::Program> ProgramsMap_t;
    typedef std::pair<std::string, cl::Program> ProgramsMapPair_t;

    cl_int m_clError;
    std::string m_clBuildErrorInfo;

    std::vector<cl::Platform> m_platforms;
    cl::Platform m_clPlatform;

    cl_context_properties m_ctxProps[3];
    cl::Context m_clContext;

    std::vector<cl::Device> m_clDevices;
    cl::Device m_clDevice;
    std::string m_baseKernelsPath;

    cl_command_queue_properties m_clCommandQueueProps;
    cl::CommandQueue m_clDefaultCommandQueue;

    ProgramsMap_t m_programs;
};

#endif // DEVICE_CONTEXT_H

