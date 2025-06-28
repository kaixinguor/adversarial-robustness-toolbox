new_path=$1
old_path=$2
diff -rq . $1 $2 --exclude='*.png' --exclude=.git --exclude=__pycache__ --exclude='*.egg-info' --exclude=results --exclude=dataset --exclude=output
