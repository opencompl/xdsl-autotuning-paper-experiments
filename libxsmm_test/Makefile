I := 16
J := 16
K := 16

a.out: main.o xsmm.o
	gcc -O3 $^ -o $@ 

main.o: main.c
	gcc -O3 -march=native -c $< -o $@ -DI=$(I) -DJ=$(J) -DK=$(K)

xsmm.o: xsmm.c
	gcc -O3 -march=native -c $< -o $@

xsmm.c:
	libxsmm_gemm_generator \
		dense \
		$@ \
		libxsmm_matmul \
		$(I) $(J) $(K) \
		$(I) $(K) $(I) \
		1 \
		0 \
		0 0 \
		hsw \
		nopf \
		DP
	sed -i '/libxsmm_num_total_flops/d' $@

clean:
	rm -f a.out xsmm.c xsmm.o main.o
