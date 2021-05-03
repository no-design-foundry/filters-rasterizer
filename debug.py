import concurrent.futures
import time


class Test:
    def __init__(self):
        pass

    def test(self, length):
        time.sleep(2)
        return length

    def test_2(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.test, 2-index/10) for index in range(20)]
            for future in concurrent.futures.as_completed(results):
                print(future.result())

test = Test()
test.test_2()