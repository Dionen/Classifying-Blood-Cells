<h1><a id="Final_Report_0"></a>Final Report</h1>
<p>Diagnosis of blood-related diseases often require labeling and characterizing elements of the patient’s blood sample. Given a microscopic picture of blood, the algorithms must identify, count and label the cells. The output differentiates the red blood cells, white blood cells and platelets utilizing image segmentation methods.</p>
<h3><a id="Dataset_5"></a>Dataset</h3>
<p>All microscopic images and annotations were provided by the following public dataset: <a href="https://github.com/Shenggan/BCCD_Dataset">https://github.com/Shenggan/BCCD_Dataset</a><br></p>
<p align="center"><img src="https://snag.gy/wquRKx.jpg" alt="Dataset Example"><br></p>
<p>At first glance:</p>
<ul>
<li>Both white blood cells (WBC) and platelets are blue/purple, while red blood cells have a red hue.</li>
<li>Platelets are the smallest elements in the picture, while WBCs are much bigger.</li>
</ul>
<h2><a id="Extracting_and_labelling_white_blood_cells_12"></a>Extracting and labelling white blood cells</h2>
<p>Color segmentation!</p>
<ul>
<li>Conversion from RGB to HSV.</li>
<li>Mask specific range of values.</li>
<li>Series of morphological operations.</li>
</ul>
<p align="center"><img src="https://snag.gy/48K2zn.jpg" alt="Initial Segmentation"></p>
<p>Needs polish for cases when two WBCs are too close too each other and would end up being counted as only one.<br>
<img src="https://snag.gy/6YUkFI.jpg" alt="Segmentation Problem"><br>
One way to solve it:</p>
<ul>
<li>Distance transforming WBC’s mask.</li>
<li>Applying certain threshold to transformed mask leaves us with only the center of each WBC.</li>
<li>Now we know the correct number of WBCs in the picture.</li>
</ul>
<p align="center"><img src="https://snag.gy/JGTzmv.jpg" alt="Problem Solution"></p>
<p>Applying watershed algorithm in order to identify the complete region occupied by each WBC defined by thresholded centers:</p>
<ul>
<li>Bind WBC mask with threshold mask to obtain “unknown region” mask.</li>
<li>The watershed algorithm expands each “center” of the mask, resulting in a good approximation of area for each element.</li>
</ul>
<p align="center"><img src="https://snag.gy/mbrAxU.jpg" alt="WBC Watershed"></p>
<h3><a id="Partial_result_34"></a>Partial result:</h3>
<p align="center"><img src="https://snag.gy/akJRYw.jpg" alt="Partial WBC"></p>
<h2><a id="Extracting_and_labelling_platelets_37"></a>Extracting and labelling platelets</h2>
<p>Platelets and WBCs are not so different in terms of color, so they utilize a similar mask.</p>
<ul>
<li>Conversion from RGB to HSV.</li>
<li>Mask specific range of values.</li>
<li>Series of morphological operations.</li>
</ul>
<p align="center"><img src="https://snag.gy/4AzfvC.jpg" alt="Initial Platelets"></p>
<ul>
<li>To avoid areas of WBCs that are caught by the mask due to similarity, all that need to be done is a subtraction of the already calculated WBC mask.</li>
<li>This is done before the morphological transformations, so as not to let dilated parts of WBC reminiscents appear.</li>
</ul>
<p align="center"><img src="https://snag.gy/xZHCnB.jpg" alt="Platelet Problem Solved"></p>
<p>Similar distance, threshold, and watershed algorithms that were applied to WBCs are also applied to the platelets.<br></p>
<p align="center"><img src="https://snag.gy/7CaPMb.jpg" alt="Similar Platelets"></p>
<h3><a id="Partial_result_53"></a>Partial result:</h3>
<p align="center"><img src="https://snag.gy/I5k0HY.jpg" alt="Final Platelets"></p>
<h2><a id="Extracting_and_labelling_red_blood_cells_55"></a>Extracting and labelling red blood cells</h2>
<p>Different masks, but same beginning.</p>
<ul>
<li>Conversion from RGB to HSV.</li>
<li>Mask specific range of values.</li>
<li>Series of morphological operations.</li>
</ul>
<p align="center"><img src="https://snag.gy/aBWvSp.jpg" alt="Initial RBC"></p>
<p>Problem!</p>
<ul>
<li>The same process from now on won’t work.</li>
<li>Due to a lack of constant shape, the algorithms will separate the mask into regions, not individuals cells.</li>
</ul>
<p align="center"><img src="https://snag.gy/mICgAc.jpg" alt="RBC Problem"></p>
<p>Solution: Applying Circle Hough Transform</p>
<ul>
<li>Besides not being constant in shape, they are still, for the mos part, oval.<br>
CHT is a feature extraction technique for detecting circles.</li>
</ul>
<p align="center"><img src="https://snag.gy/xoEvi5.jpg" alt="CHT RBC"></p>
<p>After computing each circle and utilizing the already calculated RBC mask to determine if it is corresponding to a possible blood cell, the following outcome is generated:</p>
<p align="center"><img src="https://snag.gy/MW6Fkj.jpg" alt="Final RBC"></p>
<h2><a id="Complete_result_79"></a>Complete result</h2>
<p align="center"><img src="https://snag.gy/oHOmhS.jpg" alt="Complete Result"></p>
<p>The Circle Hough Transform works pretty well for the problem, that doesn't have an exact solution. The platelets are the main inaccuracy in this implementation due to most times being faded and blurred, which goes undetect by the masks sometimes. General accuracy of 87,6%, considering total number WBC, RBC and Platelets found compared to the dataset annotation files.</p>
<blockquote>
<p>WBC:   94.4%<br>
RBC:     87.5%<br>
Platelets:   82.0%</p>
</blockquote>
