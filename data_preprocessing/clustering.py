import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def elbow_plot(self,data):
        
        self.logger_object.log(self.file_object,'Enter the elbow method of the KMeansClustering class')
        wcss=[]
        try:
            for i in range(1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')

            self.kn=KneeLocator(range(1,11),wcss,curve='convex',direction='decreasing')
            self.logger_object.log(self.file_object,'The optimum number of clusters is :'+str(self.kn.knee)+'.Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception Message :'+str(e))
            self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of KMeansClustering class')
            raise Exception()


    def create_clusters(self,data,number_of_clusters):

        self.logger_object.log(self.file_object,'Entered the create_clusters method of the KMeansClustering class')
        self.data=data
        try:
            self.kmeans=KMeans(n_clusters=number_of_clusters,init='k-means++',random_state=42)
            self.y_kmeans=self.kmeans.fit_predict(data)
            self.file_op=file_methods.File_Operation(self.file_object,self.logger_object)
            self.save_model=self.file_op.save_model(self.kmeans,'KMeans')
            
            self.data['Cluster']=self.y_kmeans
            self.logger_object.log(self.file_object,'successfully created '+str(self.kn.knee)+ ' clusters. Exited the create_cluster method of KMeansClustering class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_cluster method of KMeansClustering class. Exception Message :'+str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()





            