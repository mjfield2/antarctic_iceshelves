#!/usr/bin/env bash

python iceshelves/abbot/abbot_inversions.py -n 100
python iceshelves/george/george_inversions.py -n 100
python iceshelves/getz/getz_inversions.py -n 100
python iceshelves/larsen/larsen_inversions.py -n 100 -f
python iceshelves/maudeast/maudeast_inversions.py -n 100 -f
python iceshelves/maudwest/maudwest_inversions.py -n 100 -f
python iceshelves/salzberger/salz_inversions.py -n 100
python iceshelves/shackleton/shack_inversions.py -n 100
python iceshelves/totten/totten_inversions.py -n 100
