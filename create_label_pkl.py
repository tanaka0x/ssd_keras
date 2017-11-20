import pickle
import argparse
from PASCAL_VOC.get_data_from_XML import XML_preprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml_dir', help='input directory that contains annotation xml files', type=str)
    parser.add_argument('output_file_path', help='path to output data', type=str)

    args = parser.parse_args()

    data = XML_preprocessor(args.input_xml_dir + '/').data

    print(f'dump to {args.output_file_path}')
    pickle.dump(data, open(args.output_file_path, 'wb'))
