class decorator_class(object):
    def __init__(self, original_function):
        self.original_function = original_function

    def __call__(self, *args, **kwargs):
        print('Call method executed before this {}'.format(self.original_function.__name__))
        return self.original_function(*args, **kwargs)




@decorator_class
def display():
    print('LOST AND FOUND :))')


@decorator_class
def display_info(name, age):
    print("{} is {} years old today!".format(name, age))


display_info("Johnnie", 45)