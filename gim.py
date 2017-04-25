#Gim for active users:
def tr(i):
    count = df.loc[df['user_id']==i]
    total=0
    for k in count['rating']:
        total = total + k 
    return total
#Genre rating of Genre Gj for user ui.
def gr(movies):
    gr1 = np.zeros(19)
    for i in range(0,19):
        

def gim_final(i,j):
    #get movies rated by user ui
    user_ui_movies = df.loc[df['user_id']==i]
    tf = user_ui_movies.shape[0]
    print tf
    tr = 0
    for k in user_ui_movies['rating']:
        tr = tr + k
    
    gr_array = np.zeros(19)
    k=0
    gr_array = gr(user_ui_movies)
