import sys

class CustomException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        self.error_msg = error_msg
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        
        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        return f"Error occured in python script name ==> [{self.file_name}] line number ==> [{self.line_no}] error message ==> [{self.error_msg}]"