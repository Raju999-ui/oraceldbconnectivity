import mysql.connector
#1.connection
con=mysql.connector.connect(host="localhost",user="root",passwd="Varshitha",database="hospital")

#2 cursor
cur=con.cursor()

#3 execute the query
r=cur.execute("select * from paatient")
l=cur.fetchall()
#d=r.description

'''for i in d:
    print(i[0],end="\t\t")
print()'''


for i in l:
    for j in i:
        print(j,end="\t\t")
    print()
#print(r)
#4 close the cursor
cur.close()

#5
con.close()
