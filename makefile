in=

run:
	cython3 --embed ${in} -o temp.c
	gcc -O3 $$(pkg-config --libs --cflags python3) temp.c -o temp
	@rm -rf temp.c
	./temp

clean:
	rm -rf temp

