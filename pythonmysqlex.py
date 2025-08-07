import mysql.connector
#1.connection
con=mysql.connector.connect(host="localhost",user="root",passwd="Varshitha",database="hospital")

#2 cursor
cur=con.cursor()

#3 execute the query
#cur.execute("create table paatient(pid int primary key,pname varchar(100))")
pid=int(input("enter id:"))
pname=input("enter name:")
q=f"insert into paatient(pid,pname) values({pid},'{pname}')";
try:
    #cur.execute(q)
    con.commit()
except:
    print("patient must be unique")
else:
    print("Data inserted successfully")
patientdetails=cur.execute("select * from paatient")
print(patientdetails)

for i in patientdetails:
    print(i)

#4 close the cursor
cur.close()

#5
con.close()
