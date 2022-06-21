from data_generation.tasks import *

TASKS=[
    # The images contain objects of the same shape.
    ['task_shape', task_shape],
    # The images contain objects in the same position.
    ['task_pos', task_pos],
    # The images contain objects of the same size.
    ['task_size', task_size],
    # The images contain objects of the same color.
    ['task_color', task_color],
    # The images contain objects of the same shape with different rotations.
    ['task_rot', task_rot],
    # The images contain objects of the same shape with different flips.
    ['task_flip', task_flip],
    # The images contain the same number of objects.
    ['task_count', task_count],
    # The images contain an object inside another object.
    ['task_inside', task_inside],
    # The images contain an object in contact with another object.
    ['task_contact', task_contact],
    # The images contain 2 objects in a rotational symmetry around a center indicated by a third small object.
    ['task_sym_rot', task_sym_rot],
    # The images contain 2 objects in a mirror symmetry.
    ['task_sym_mir', task_sym_mir],
    # The images contain a set of objects that have the same spatial configuration.
    ['task_pos_pos_1', task_pos_pos_1],
    # Each image contains 2 sets of objects that have the same spatial configuration.
    ['task_pos_pos_2', task_pos_pos_2],
    # The images contain a sets of objects that are aligned. The number of objects in the set is the same.
    ['task_pos_count_2', task_pos_count_2],
    # In each image, the relationship between the number of objects on one side and the number of objects on the other side is the same. For example, if the number of objects on the left is always bigger than the number of objects on the right. 
    ['task_pos_count_1', task_pos_count_1],
    # The images contain the same number of objects. One of the objects will maintain the same order over both spatial dimensions across images.
    ['task_pos_pos_4', task_pos_pos_4],
    # The images contain sets of aligned objects. The number of sets is the same in all images.
    ['task_pos_count_3', task_pos_count_3],
    # The images contain the same number of objects that contain an object.
    ['task_inside_count_1', task_inside_count_1],
    # The number of objects is either odd or even in all images.
    ['task_count_count', task_count_count],
    # Each image contain 2 similarly shaped objects.
    ['task_shape_shape', task_shape_shape],
    # Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images.
    ['task_shape_contact_2', task_shape_contact_2],
    # In each image, there is one group of connected objects.
    ['task_contact_contact_1', task_contact_contact_1],
    # Each image contains 3 objects. In all images, 1 object contains the 2 other objects.
    ['task_inside_inside_1', task_inside_inside_1],
    # In each image, one object contains all the other objects.
    ['task_inside_inside_2', task_inside_inside_2],
    # Objects that contain objects are always on the same side of the image.
    ['task_pos_inside_3', task_pos_inside_3],
    # In each image, an object contains another object. The inner object is always on the same side of the outer object. 
    ['task_pos_inside_1', task_pos_inside_1],
    # In each image, an object contains 2 objects. The 2 inner objects are always positioned similarly within the outer object.  
    ['task_pos_inside_2', task_pos_inside_2],
    # In each image, the object that contains another object is always positioned similarly with respect to the other objects across the same dimension.
    ['task_pos_inside_4', task_pos_inside_4],
    # Each image contain 2 similarly shaped objects with different rotations.
    ['task_rot_rot_1', task_rot_rot_1],
    # Each image contain 2 similarly shaped objects flipped differently.
    ['task_flip_flip_1', task_flip_flip_1],
    # Each image contain 2 similarly shaped objects, the difference between rotation of the 2 objects is the same in all images.
    ['task_rot_rot_3', task_rot_rot_3],
    # Each image contains 2 pairs of objects. Objects of each pair have the same position in one spatial dimension (for example the x axis) but different positions in the other dimension (for example the y axis). The distance between 2 objects of a pair along the second dimension and the position of the pair along the first dimension maintain the same order across images. 
    ['task_pos_pos_3', task_pos_pos_3],
    # The images contain 3 sets of aligned objects. The sum of the numbers of objects in 2 sets equals the number of objects in the third set.
    ['task_pos_count_4', task_pos_count_4],
    # Each image contains 2 objects. The first object is smaller than the second object in all images.
    ['task_size_size_1', task_size_size_1],
    # Each image contains 3 objects. 2 objects of each image have similar sizes and the other object has a different size.
    ['task_size_size_2', task_size_size_2],
    # Each image contains 3 objects. The 3 objects of each image have different sizes.
    ['task_size_size_3', task_size_size_3],
    # Each image contains 2 objects. In each image, one object's size is half the other object's size.
    ['task_size_size_4', task_size_size_4],
    # Each image contains 3 pairs of aligned objects. The object sizes in each pair are maintained in all images.
    ['task_size_size_5', task_size_size_5],
    # Each image contains objects in a mirror symmetry with respect to the same axis. The distance to the axis is correlated with the size for each object.
    ['task_size_sym_1', task_size_sym_1],
    # Each image contains objects in a rotational symmetry with respect to the center of the image. The distance to the center is correlated with the size for each object.
    ['task_size_sym_2', task_size_sym_2],
    # Each image contains 2 objects with similar color hues.
    ['task_color_color_1', task_color_color_1],
    # All objects in an image have the same hue.
    ['task_color_color_2', task_color_color_2],
    # Each image contains 2 objects that have a mirror symmetry. The axis of symmetry is the same in all images.
    ['task_sym_sym_1', task_sym_sym_1],
    # Each image contains 2 objects that have a rotational symmetry. The angle of symmetry is the same in all images.
    ['task_sym_sym_2', task_sym_sym_2],
    # Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images. The objects have different sizes and colors.
    ['task_shape_contact_3', task_shape_contact_3],
    # Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images. The objects have different colors.
    ['task_shape_contact_4', task_shape_contact_4],
    # In each image, each object is in contact with another object.
    ['task_contact_contact_2', task_contact_contact_2],
    # In each image, the large object is always positioned similarly with respect to the small object along the same dimension.
    ['task_pos_size_1', task_pos_size_1],
    # In each image, the large object is always positioned similarly with respect to the small object along the same dimension.
    ['task_pos_size_2', task_pos_size_2],
    # In each image, the object with the first shape is always positioned similarly with respect to the other object along the same dimension.
    ['task_pos_shape_1', task_pos_shape_1],
    # Each image contains many pairs of close objects. The pairs, identified by the object shapes, are similar across images.
    ['task_pos_shape_2', task_pos_shape_2],
    # The images contain a set of objects that have the same spatial configuration. The spatial configurations are rotated and objects in the configuration are rotated with the same angle.
    ['task_pos_rot_1', task_pos_rot_1],
    # Each image contains 2 sets of objects that have the same spatial configuration. One of the spatial configurations is rotated and objects in the configuration are rotated with the same angle.
    ['task_pos_rot_2', task_pos_rot_2],
    # In each image, the objects are always positioned similarly along the same dimension with respect to their colors.
    ['task_pos_col_1', task_pos_col_1],
    # The images contain a set of objects that have the same spatial configuration and color configuration. Both configurations are maintained in all images.
    ['task_pos_col_2', task_pos_col_2],
    # The images contain two objects in contact along the same direction.
    ['task_pos_contact', task_pos_contact],
    # The images contain two pairs of objects. Each pair of objects have the same size and shape.
    ['task_size_shape_1', task_size_shape_1],
    # The images contain 4 objects with 4 different shapes (a,b,c,d). When a is bigger than b, c is bigger d. Conversly, when a is smaller than b, c is smaller than d. 
    ['task_size_shape_2', task_size_shape_2],
    # Each image contain 3 objects with the same shape. The smallest is rotated to the left and the largest is rotated to the right with respect to the average one.
    ['task_size_rot', task_size_rot],
    # In each image, one of the objects contains an object. The object that contains an object is either always the larger one or always the smaller one.
    ['task_size_inside_1', task_size_inside_1],
    # The images contain 4 objects a,b,c and d. The contact and size configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. A size configuration is A and B having the same size which is bigger than C and D.
    ['task_size_contact', task_size_contact],
    # Each image contains sets of objects. Within each set, objects have the same size. All images have the same number of objects in each set. 
    ['task_size_count_1', task_size_count_1],
    # Each image contains sets of objects. Within each set, objects have the same size. All images have the same number of sets. 
    ['task_size_count_2', task_size_count_2],
    # Each image contains 2 objects. The association between shapes and colors is fixed across images. 
    ['task_shape_color', task_shape_color],
    # In each image, objects with the same shapes have the same color. 
    ['task_shape_color_2', task_shape_color_2],
    # In each image, objects with the same shapes have different colors. 
    ['task_shape_color_3', task_shape_color_3],
    # Each image contains 2 objects, A and B, whose shapes don't change across images. In each image, A is inside B.
    ['task_shape_inside', task_shape_inside],
    # Each image contains several objects. In each image, if an object is inside another object, they share the same shape.
    ['task_shape_inside_1', task_shape_inside_1],
    # Each image contains sets of objects. Within each set, objects have the same shape. The number of objects in each set is constant across images.
    ['task_shape_count_1', task_shape_count_1],
    # Each image contains sets of objects. Within each set, objects have the same shape. The number of sets is constant across images.
    ['task_shape_count_2', task_shape_count_2],
    # In each image objects with the same shapes have the same color. All objects are randomly rotated. 
    ['task_rot_color', task_rot_color],
    # Each image contains 2 pairs of objects. In each pair, one of the objects is inside the other. One of the pairs is a rotation of the other pair.
    ['task_rot_inside_1', task_rot_inside_1],
    # In each image, if an object contains on object, they share the same shape. The objects are randomly rotated.
    ['task_rot_inside_2', task_rot_inside_2],
    # Each image contains sets of objects. Within each set objects have the same shape and are randomly rotated. The number of objects in each set is constant across images.
    ['task_rot_count_1', task_rot_count_1],
    # Each image contains 2 objects, A and B, whose colors don't change across images. In each image, A is inside B.
    ['task_color_inside_1', task_color_inside_1],
    # Each image contains several objects. In each image, if an object is inside another object, they share the same color.
    ['task_color_inside_2', task_color_inside_2],
    # The images contain 4 objects A,B,C and D. The contact and color configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. A color configuration is A and B having the same color which is different from the colors of C and D.
    ['task_color_contact', task_color_contact],
    # Each image contains sets of objects. Within each set, objects have the same color. The number of objects in each set is constant across images.
    ['task_color_count_1', task_color_count_1],
    # Each image contains sets of objects. Within each set, objects have the same color. The number of sets is constant across images.
    ['task_color_count_2', task_color_count_2],
    # The images contain 4 objects A,B,C and D, 2 of which contain an object. The insideness and contact configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. An insideness configuration is A and B containing an object and C and D containing no objects.
    ['task_inside_contact', task_inside_contact],
    # Each image contains sets of objects. Within each set, objects are in contact. The number of objects in each set is constant across images.
    ['task_contact_count_1', task_contact_count_1],
    # Each image contains sets of objects. Within each set, objects are in contact. The number of sets is constant across images.
    ['task_contact_count_2', task_contact_count_2],
    # Each image contains 2 objects. The larger object always has the same color.
    ['task_size_color_1', task_size_color_1],
    # In each image, objects with the same size have the same color.
    ['task_size_color_2', task_size_color_2],
    # In each image, each pair of objects are in a rotational symmetry around the center of the image. The objects of each pair have the same color.
    ['task_color_sym_1', task_color_sym_1],
    # In each image, each pair of objects are in a mirror symmetry around the same axis. The objects of each pair have the same color.
    ['task_color_sym_2', task_color_sym_2],
    # Each image contains 2 objects that have similar shapes and a third differently shaped object. The similarly shaped objects are randomly rotated. The shapes are the same across images. 
    ['task_shape_rot_1', task_shape_rot_1],
    # In each image, similarly shaped objects are in contact.
    ['task_shape_contact_5', task_shape_contact_5],
    # Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact and randomly rotated in all images.
    ['task_rot_contact_1', task_rot_contact_1],
    # In each image, similarly shaped objects are in contact. All objects are randomly rotated.
    ['task_rot_contact_2', task_rot_contact_2],
    # In each image, only objects that contain other objects are in mirror symmetry over the same axis.
    ['task_inside_sym_mir', task_inside_sym_mir],
    # Each image contains sets of objects. Within each set objects have the same shape and are randomly flipped. The number of objects in each set is constant across images.
    ['task_flip_count_1', task_flip_count_1],
    # Each image contains 2 pairs of objects. In each pair, one of the objects is inside the other. One of the pairs is a flip of the other pair.
    ['task_flip_inside_1', task_flip_inside_1],
    # In each image, if an object contains on object, they share the same shape. The objects are randomly flipped.
    ['task_flip_inside_2', task_flip_inside_2],
    # In each image objects with the same shapes have the same color. All objects are randomly flipped. 
    ['task_flip_color_1', task_flip_color_1],
    # Each image contains 2 objects that have similar shapes and a third differently shaped object. The similarly shaped objects are randomly flipped. The shapes are the same across images. 
    ['task_shape_flip_1', task_shape_flip_1],
    # Each image contains 4 objects. All 4 objects have the same shape and are placed in the same 4 locations. Objects aligned vertically have a the same difference in rotation angle and objects aligned vertically are flipped differently.
    ['task_rot_flip_1', task_rot_flip_1],
    # Each image contain 3 objects with the same shape. The smallest object is vertically flipped and the largest object is horizantally flipped with respect to the average one.
    ['task_size_flip_1', task_size_flip_1],
    # Each image contain 3 objects with the with successive positions along a spatial dimension. The rotation angles remain the same for each position across images.
    ['task_pos_rot_3', task_pos_rot_3],
    # Each image contain 3 objects with the with successive positions along a spatial dimension. The flips remain the same for each position across images.
    ['task_pos_flip_1', task_pos_flip_1],
    # Each image contains 2 objects with similar shapes. One of the objects is flipped along the axis formed by the two objects.
    ['task_pos_flip_2', task_pos_flip_2],
    # Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact and randomly flipped in all images.
    ['task_flip_contact_1', task_flip_contact_1],
    # In each image, similarly shaped objects are in contact. All objects are randomly flipped.
    ['task_flip_contact_2', task_flip_contact_2],
]















































































































































































