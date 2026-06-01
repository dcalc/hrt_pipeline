from sophi_hrt_pipe.hrt_hmi_wcs_correction import run_HMI_correction
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filename (with path)')
    parser.add_argument('filename', type=str, help='The name of the file to correct or directory with wildcard')
    parser.add_argument('-i', '--hmi_path', type=str, default=None, help='HMI directory file input')
    parser.add_argument('-v','--verbose', action='store_true',help='plot corrected images if true')
    parser.add_argument('-p','--print_values', action='store_true',help='print results if true')
    parser.add_argument('-d','--drms', action='store_true',help='use local DRMS if true')
    parser.add_argument('-c', '--crota', type=float, default=0.15, help='manual correction to CROTA')
    
    args = parser.parse_args()
    
    verbose = args.verbose
    filename = args.filename
    print_values = args.print_values
    local_drms = args.drms
    hmi_path = args.hmi_path# if args.hmi_path is not None else None
    crota_manual_correction = args.crota

    # filename can be a string with wildcard or directory path
    run_HMI_correction(0, 0, verbose = verbose, filename=filename, print_values=print_values, hmi_path=hmi_path, crota_manual_correction=crota_manual_correction, local_drms = local_drms)