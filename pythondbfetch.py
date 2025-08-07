import oracledb

#step1 establish a connection between python and oracle

con = oracledb.connect(
    user='system',
    password='raj123',
    dsn='localhost/xe'
    
)
cur=con.cursor()

r=cur.execute("select * from student")

#print(r,type(r))
'''print(r.fetchone())
print(r.fetchone())
print(r.fetchone())
print(r.fetchone())
print(r.fetchone())
print(r.fetchone())
'''
#print(r.fetchmany(3))
l=r.fetchall()
d=r.description

print("-"*50)
#rint(d,type(d))
for i in d:
   print(i[0],end="\t\t")
print()
   
#print(l)

for i in l:
    for j in i:
     print(j,end="\t\t")
    print()
print("-"*50)

