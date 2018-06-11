import argparse

# パーサーを作る
parser = argparse.ArgumentParser(
    prog='argparseTest.py',  # プログラム名
    usage='Demonstration of argparser',  # プログラムの利用方法
    description='description',  # 引数のヘルプの前に表示
    epilog='end',  # 引数のヘルプの後で表示
    add_help=True,  # -h/–help オプションの追加
)

# 引数の追加
parser.add_argument('-v', '--verbose', help='select mode',
                    type=int,
                    choices=[0, 1])  # 引数名に"-"をつけるとoptional argumentになる
parser.add_argument('-i', help='integer',
                    type=int, required=True)

# 引数を解析する
args = parser.parse_args()

if args.verbose == 0:
    if args.i % 2 == 1:
        print(str(args.i) + ' : Odd')
    else:
        print(str(args.i) + ' : Even')
elif args.verbose == 1:
    if args.i % 2 == 1:
        print(str(args.i) + ' : 奇数')
    else:
        print(str(args.i) + ' : 偶数')
