tail -f UnitySDK.log | unbuffer -p grep floor: | unbuffer -p cut -f 2 -d ':'  | unbuffer -p sed -E 's/ /floor=/g' | unbuffer -p grep -E '[0-9]'
