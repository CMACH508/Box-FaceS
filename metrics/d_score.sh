list="0 1 2 3 4 5"
for i in $list;
do
python d_score.py --index $i --edit <EDITED RESULTS> --src <RECONSTRUCTION>;
done

