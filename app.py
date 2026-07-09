import gradio as gr

from src.perceul.selectors import NumericSelector
from src.perceul.clustering import explore_clusters, final_clustering
from src.perceul.interpretation import get_pca_loadings

#========== GRADIO INTERFACE ==========
with gr.Blocks(title="PERCEUL: Perception-Based Worker Profiler") as app:

    gr.Markdown("# 🧠 PERCEUL: Profiler of Perception and Cognitive Ergonomics in the Workplace")

    file_input = gr.File(label="Upload CSV")

    # invisible states to hold PCA model and feature names =======
    pca_state = gr.State()
    feature_names_state = gr.State()
    # ============================================================

    with gr.Tab("Cluster Exploration"):
        
        btn = gr.Button("Explore Clusters")
        
        plot_output = gr.Plot()
        cluster_summary_output = gr.JSON(label="Cluster Sizes")
        outlier_output = gr.Markdown("### Exploration Report")
    
    with gr.Tab("Final Clustering"):

        top_features = gr.Number(
            value=5, 
            label="Number of Features to Display", 
            minimum=3, 
            maximum=10,
            step=1,
            precision=0
        )
    
        run_btn = gr.Button("Run Final Clustering")
        best_k_out = gr.Number(label="Number of Clusters `K` (Read-Only)", interactive=False, precision=0)
        gr.Markdown("### Cluster Characteristics")

        with gr.Row():
            with gr.Column(scale=2):
                pca_plot_out = gr.Plot()
            with gr.Column(scale=3):
                deviations_out = gr.Markdown()
            
        with gr.Row():
            loadings_out = gr.Dataframe(label="PCA Loadings")   

    
    # ==========================================================
    #                   BUTTON CALLBACKS
    # ==========================================================

    btn.click(
        fn=explore_clusters,
        inputs=[file_input],
        outputs=[plot_output, cluster_summary_output, outlier_output]
        )
    
    run_btn.click(
        fn=final_clustering,
        inputs=[file_input, top_features],
        outputs=[pca_plot_out, best_k_out, deviations_out, pca_state, feature_names_state]
    ).then(
        fn=get_pca_loadings,
        inputs=[pca_state, feature_names_state],
        outputs=[loadings_out]
    )

if __name__ == "__main__":
    app.launch(share=False, debug=True)