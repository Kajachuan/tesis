import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitad1", type=str, help="Directorio de la primera mitad")
    parser.add_argument("--mitad2", type=str, help="Directorio de la segunda mitad")
    parser.add_argument("--output", type=str, help="Directorio donde se guarda el resultado")
    args = parser.parse_args()

    f1 = open(args.mitad1)
    f2 = open(args.mitad2)

    data1 = json.load(f1)
    data2 = json.load(f2)

    print(data1)

    f1.close()
    f2.close()

if __name__ == '__main__':
    main()
