cmake_minimum_required(VERSION 3.21)

project(
	nullfighter
	LANGUAGES    C ASM
	VERSION      0.1.0
	DESCRIPTION  "Skinned FBX character viewer for PlayStation 1"
)

# Convert both characters into the runtime formats.  Requires python3 with
# numpy + Pillow (+ xatlas for --reatlas) on the build host.
#   schoolgirl  un-rigged Tripo mesh: decimated to MODEL_TRIS triangles,
#               re-unwrapped UVs, auto rig + procedural animations
#   character   skinned + animated FBX
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ASSET_SRC   ${PROJECT_SOURCE_DIR}/assets/src)
set(ASSET_OUT   ${PROJECT_BINARY_DIR}/assets)
set(MODEL_TRIS "1400" CACHE STRING "schoolgirl: target triangle count (0 = no decimation)")
set(TOOL_DEPS ${PROJECT_SOURCE_DIR}/tools/fbx2ps1.py ${PROJECT_SOURCE_DIR}/tools/fbxparse.py
              ${PROJECT_SOURCE_DIR}/tools/meshopt.py ${PROJECT_SOURCE_DIR}/tools/autorig.py)

# convert_model(name fbx texture [FACE] extra-args...): FACE also writes
# <name>_face.tim (second texture page for the face)
function(convert_model name fbx tex)
	set(bin ${ASSET_OUT}/${name}.bin)
	set(tim ${ASSET_OUT}/${name}.tim)
	set(args ${ARGN})
	set(outputs ${bin} ${tim})
	set(face_args)
	list(FIND args FACE face_index)
	if(NOT face_index EQUAL -1)
		list(REMOVE_AT args ${face_index})
		set(face_args --face-tex --out-face-tim ${ASSET_OUT}/${name}_face.tim)
		list(APPEND outputs ${ASSET_OUT}/${name}_face.tim)
	endif()
	add_custom_command(
		OUTPUT  ${outputs}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSET_OUT}
		COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/fbx2ps1.py
		        ${fbx} --texture ${tex} ${args} ${face_args} --out-bin ${bin} --out-tim ${tim}
		DEPENDS ${TOOL_DEPS} ${fbx} ${tex}
		COMMENT "Converting ${name} -> ${name}.bin / ${name}.tim"
		VERBATIM
	)
	set(MODEL_OUTPUTS ${MODEL_OUTPUTS} ${outputs} PARENT_SCOPE)
endfunction()

convert_model(schoolgirl ${ASSET_SRC}/schoolgirl/schoolgirl.fbx ${ASSET_SRC}/schoolgirl/basecolor.jpg
              FACE --target-tris ${MODEL_TRIS} --reatlas --autorig
              --remove-hidden 0.02 --double-sided 0.36 0.56)
convert_model(character ${ASSET_SRC}/character.fbx ${ASSET_SRC}/character.fbm/basecolor.jpg --front=-z)
add_custom_target(assets DEPENDS ${MODEL_OUTPUTS})

psn00bsdk_add_executable(nullfighter GPREL
	src/main.c
	src/anim.c
	src/render.c
	src/camera.c
	src/ik.c
)
target_include_directories(nullfighter PRIVATE src)
target_link_libraries(nullfighter PRIVATE psxgpu psxgte psxetc psxapi)
add_dependencies(nullfighter assets)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_bin ${ASSET_OUT}/schoolgirl.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_tim ${ASSET_OUT}/schoolgirl.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_face_tim ${ASSET_OUT}/schoolgirl_face.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE character_bin  ${ASSET_OUT}/character.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE character_tim  ${ASSET_OUT}/character.tim)

psn00bsdk_add_cd_image(
	iso
	nullfighter
	iso.xml
	DEPENDS nullfighter system.cnf
)
