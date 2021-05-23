from concurrent.futures import ThreadPoolExecutor
import threading
import time

undo_set = set()
# 定义一个准备作为线程任务的函数
def action(max):
  my_sum = 0
  for i in range(max):
    print(threading.current_thread().name + ' ' + str(i))
    my_sum += i
  time.sleep(max)
  return my_sum
# 创建一个包含2条线程的线程池
with ThreadPoolExecutor(max_workers=2) as pool:
  # 向线程池提交一个task, 50会作为action()函数的参数
  future1 = pool.submit(action, 2)
  # 向线程池再提交一个task, 100会作为action()函数的参数
  future2 = pool.submit(action, 4)
  future3 = pool.submit(action, 2)
  # 向线程池再提交一个task, 100会作为action()函数的参数
  try:
    future4 = pool.submit(action, 4)
  except Exception as e:
      print(e)
  future4.cancel()
  def get_result(future):
    print(future.result())
  # 为future1添加线程完成的回调函数
  future1.add_done_callback(get_result)
  # 为future2添加线程完成的回调函数
  future2.add_done_callback(get_result)
  future3.add_done_callback(get_result)
#   future4.add_done_callback(get_result)
  print('--------------')


# import ctypes
# import inspect
# import threading
# import time


# def _async_raise(tid, exctype):
#     '''Raises an exception in the threads with id tid'''
#     if not inspect.isclass(exctype):
#         raise TypeError("Only types can be raised (not instances)")
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid,
#                                                   ctypes.py_object(exctype))
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         # "if it returns a number greater than one, you're in trouble,
#         # and you should call it again with exc=NULL to revert the effect"
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
#         raise SystemError("PyThreadState_SetAsyncExc failed")

# class ThreadWithExc(threading.Thread):
#     '''A thread class that supports raising exception in the thread from
#        another thread.
#     '''
#     def raiseExc(self, exctype):
#         """Raises the given exception type in the context of this thread.

#         If the thread is busy in a system call (time.sleep(),
#         socket.accept(), ...), the exception is simply ignored.

#         If you are sure that your exception should terminate the thread,
#         one way to ensure that it works is:

#             t = ThreadWithExc( ... )
#             ...
#             t.raiseExc( SomeException )
#             while t.isAlive():
#                 time.sleep( 0.1 )
#                 t.raiseExc( SomeException )

#         If the exception is to be caught by the thread, you need a way to
#         check that your thread has caught it.

#         CAREFUL : this function is executed in the context of the
#         caller thread, to raise an excpetion in the context of the
#         thread represented by this instance.
#         """
#         tid = self.ident
#         print(tid)
#         _async_raise(tid, exctype)
    
#     def run(self):
#         while True:
#             print('-------')
#             time.sleep(0.5)


# if __name__ == "__main__":
#     t = ThreadWithExc()
#     t.daemon = True
#     t.start()
#     time.sleep(2)
#     t.raiseExc(RuntimeError)
#     print(t.isAlive())
