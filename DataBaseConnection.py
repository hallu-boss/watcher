import sqlite3
import datetime


class DataBaseConnection:

    def __init__(self):
        self.conn = sqlite3.connect('dataBase_PSIO.db')

        if(self.conn == None):
            print("Connection to database failed")
            exit(-1)

        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def insertEvent(self, ID_EMPLOYEE , DESCRIPTION ):

        EVENT_TIME = datetime.datetime.now()
        query = "INSERT INTO EVENTS (ID_EMPLOYEE, EVENT_TIME, DESCRIPTION) VALUES (?,?,?)"

        self.cursor.execute(query, (ID_EMPLOYEE, EVENT_TIME, DESCRIPTION))
        self.conn.commit()

    def selectEmployee(self, ID_EMPLOYEE):
        query = "SELECT * FROM EMPLOYEE WHERE ID_EMPLOYEE = ?"

        result = self.cursor.execute(query, (ID_EMPLOYEE,))
        result = result.fetchall()
        return result

    def selectEmployeeCar(self, ID_EMPLOYEE, REG_PLATE_NO):
        query = "SELECT * FROM CARS WHERE REG_PLATE_NO = ?"



db = DataBaseConnection()

