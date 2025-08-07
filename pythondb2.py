import oracledb

#step1 establish a connection between python and oracle

con = oracledb.connect(
    user='system',
    password='raj123',
    dsn='localhost/xe'
    
)

#step2:create cursor object
cur=con.cursor()

#step3 execute queries
'''q="create table student(rno varchar(10),sname varchar(100),marks int)"

rno=input("enter rno:")
sname=input("enter student name:")
marks=int(input("enter marks:"))''' 

q1="insert into student(rno,sname,marks) values(:r,:s,:m)"

try:    
    #cur.execute("update student set sname='modi' where rno='2'")   
    #cur.execute("insert into student(rno,sname,marks) values(:r,:s,:m)",r=rno,s=sname,m=marks)
    cur.execute("delete from student where rno=2")
    con.commit()
    #cur.execute(q)
    print("student details successfully deleted")
except:    
    print("Table already exists")


#step 4 cursor object close
#cur.close()

#step5 connection close
con.close()