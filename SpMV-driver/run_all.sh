#!/bin/sh
#

for i in `seq 0 15`
do
  cpufreq-set -c${i} -g userspace
done

for frec in 2400000 2000000 1600000 1200000
do
    for i in `seq 0 15`
    do  
        cpufreq-set -c ${i} -f ${frec}
    done
    cpufreq-info | grep 'current CPU'

    # EDIT
    matrices=matrices/list_sym.txt
    #matrices=matrices/list_noSym.txt
    #matrices=matrices/listTest_sym.txt
    #matrices=matrices/listTest_noSym.txt
    sed -e "s/\t/ /g" $matrices | awk '{print "https://sparse.tamu.edu/RB/"$3"/"$2".tar.gz"}' | while read line
    do
      count=0
      nombre=$(echo $line | cut -d'/' -f6 | cut -d'.' -f1)
      fich=$(echo $line | cut -d'/' -f6)
      ruta="$nombre/$nombre.rb"
      echo "output_$nombre""_$frec.h5"
      if [ -f output_${nombre}_${frec}.h5 ]; then 
        filesize=`stat -c %s output_${nombre}_${frec}.h5`
      else
        filesize=0
      fi

      while [ "$filesize" == "0" ] && [ $count -lt 3 ]; do
        if [ ! -f ${nombre}.rb ]; then 
          wget $line
        fi 

        tar -xvzf $fich -C . --strip-components=1 $ruta
        touch "output_$nombre""_$frec.h5"
        chmod 777 "output_$nombre""_$frec.h5"
        chmod 777 "$nombre.rb"
        # EDIT
        sudo ./launch.sh "$nombre.rb" "$frec" "1"  #SUDO FOR ENERGY. Parameter 1 indicates symmetric matrices.

        filesize=`stat -c %s output_${nombre}_${frec}.h5`
        count=$(( $count + 1 ))
        if [ "$filesize" != "0" ] || [ "$count" == "3" ]; then
          rm  $fich "$nombre.rb"
        fi

      done 
    done
done
