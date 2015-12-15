//======== Copyright (c) 2008-2015, Filip Vaverka, All rights reserved. ======//
//
// Purpose:     OpenCL and wrapper error codes
//
// $NoKeywords: $OpenCLStub $error_codes.h
// $Date:       $2015-11-18
//============================================================================//

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

/**
 * @brief OpenCL wrapper error codes.
 */
enum WrapperErrorCodes_t
{
    ERR_SUCCESS							= 0,
    ERR_GENERIC_ERROR					= -1,
    ERR_GET_PLATFORMS_FAILED			= -2,
    ERR_OCL_CONTEXT_CREATE_FAILED		= -3,
    ERR_OCL_GET_DEVICES_FAILED			= -4,
    ERR_PROGRAM_FILE_OPEN_FAILED		= -5,
    ERR_OCL_PROGRAM_CREATION_FAILED		= -6,
    ERR_OCL_PROGRAM_BUILD_FAILED		= -7,
    ERR_OCL_CMDQUEUE_INIT_FAILED		= -8,
    ERR_PROGRAM_NOT_LOADED				= -9,
    ERR_KERNEL_CREATION_FAILED          = -10,
    ERR_BUFFER_CREATION_FAILED			= -11,
    ERR_BUFFER_WRITE_FAILED				= -12,
    ERR_BUFFER_READ_FAILED				= -13,
    ERR_KERNEL_ARGSET_FAILED			= -14,
    ERR_KERNEL_RUN_FAILED				= -15,
    ERR_BUFFER_FILL_FAILED				= -16,
    ERR_BUFFER_MAP_FAILED				= -17,
    ERR_BUFFER_UNMAP_FAILED				= -18,

    ERR_INVALID_PARAM					= -19,
    ERR_FILE_OPEN_FAILED				= -20,

    ERR_CODE_LAST                       = -21
};

const char *WrapperErrorCodeToString(int errorCode);
const char *OpenCLErrorCodeToString(int errorCode);

#endif // ERROR_CODES_H

