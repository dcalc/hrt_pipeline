from sophi_hrt_pipe.hrt_fdt_wcs_correction import run_FDT_correction
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filename (with path)')
    parser.add_argument('filename', type=str, help='The name of the file to correct or directory with wildcard')
    parser.add_argument('-v','--verbose', action='store_true',help='plot corrected images if true')
    parser.add_argument('-p','--print_values', action='store_true',help='print results if true')

    args = parser.parse_args()
    
    verbose = args.verbose
    filename = args.filename
    print_values = args.print_values

    # filename can be a string with wildcard or directory path
    run_FDT_correction(0, 0, verbose = verbose, filename=filename, print_values=print_values)