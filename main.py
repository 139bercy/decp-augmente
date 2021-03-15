import time
t1 = time.clock()

import nettoyage
import enrichissement

def main():
    nettoyage.main()
    enrichissement.main()


if __name__ == "__main__":
    main()

t2 = time.clock()
print((t2-t1)/60)