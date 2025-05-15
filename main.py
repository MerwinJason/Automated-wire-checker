"""
Main entry point for the Breadboard Connection Checker application.
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui import BreadboardApp
from logger_setup import logger

if __name__ == '__main__':
    logger.info("Application starting from main.py")
    app = QApplication(sys.argv)
    main_window = BreadboardApp()
    main_window.show()
    sys.exit(app.exec_())