program infer
  use, intrinsic :: iso_fortran_env, only : sp => real32
  use ftorch

  implicit none

  real(kind=sp), allocatable, dimension(:,:,:,:) :: input_data_4d
  real(kind=sp), allocatable, dimension(:,:,:,:) :: ref_data_4d
  real(kind=sp), allocatable, dimension(:,:,:,:) :: prediction_4d
  integer(kind=ftorch_int), parameter :: layout_4d(4) = [1, 2, 3, 4]

  real(kind=sp), allocatable, dimension(:,:) :: input_data_2d
  real(kind=sp), allocatable, dimension(:,:) :: ref_data_2d
  real(kind=sp), allocatable, dimension(:,:) :: prediction_2d
  integer(kind=ftorch_int), parameter :: layout_2d(2) = [1, 2]

  integer :: num_args, i
  character(len=128), dimension(:), allocatable :: args
  character(len=128) :: model_type, test_data_path, script_model_path
  character(len=128) :: prefix, path

  ! ftorch data structures
  type(torch_model) :: model
  type(torch_tensor), dimension(1) :: input_tensor, output_tensor


  logical :: success
  integer, parameter :: num_loops = 1

  num_args = command_argument_count()
  allocate(args(num_args))
  do i = 1, num_args
     call get_command_argument(i,args(i))
  end do

  ! Process data directory argument, if provided
  if (num_args == 3) then
    model_type = trim(args(1))
    test_data_path = trim(args(2))
    script_model_path = trim(args(3))
  else
    print *, "Error: please pass correct number of args."
  end if

  print *, 'model_type        = ', trim(model_type)
  print *, 'test_data_path    = ', trim(test_data_path)
  print *, 'script_model_path = ', trim(script_model_path)
  print *, '----------------------------------------------'

  prefix = ""
  if (model_type == "ann") then
    prefix = "ann-cnn"
  else if (model_type == "attention") then
    prefix = "unet"
  end if

  if (model_type == "ann") then

    path = trim(test_data_path) // "/" // trim(prefix) // "-input.nc"
    call read_dataset_from_nc_2d(input_data_2d, trim(path))
    path = trim(test_data_path) // "/" // trim(prefix) // "-predict.nc"
    call read_dataset_from_nc_2d(ref_data_2d, trim(path))
    print *, '----------------------------------------------'

    allocate(prediction_2d, mold=ref_data_2d)

    ! Create Torch input/output tensors from the above arrays
    call torch_tensor_from_array(input_tensor(1), input_data_2d, layout_2d, torch_kCUDA)
    call torch_tensor_from_array(output_tensor(1), prediction_2d, layout_2d, torch_kCPU)

    ! Load ML model
    call torch_model_load(model, trim(script_model_path) // "/nlgw_ann_gpu_scripted.pt", device_type=torch_kCUDA, device_index=0)

    ! Infer
    do i = 1, num_loops
      call torch_model_forward(model, input_tensor, output_tensor)
    end do

    print *, 'max diff = maxval(abs(prediction - ref_data))'
    print *, maxval(abs(prediction_2d - ref_data_2d))

    success = assert_allclose_real32_2d(prediction_2d, ref_data_2d, "check", 1e-5)
  else if (model_type == "attention") then

    path = trim(test_data_path) // "/" // trim(prefix) // "-input.nc"
    call read_dataset_from_nc_4d(input_data_4d, trim(path))
    path = trim(test_data_path) // "/" // trim(prefix) // "-predict.nc"
    call read_dataset_from_nc_4d(ref_data_4d, trim(path))
    print *, '----------------------------------------------'

    allocate(prediction_4d, mold=ref_data_4d)

    ! Create Torch input/output tensors from the above arrays
    call torch_tensor_from_array(input_tensor(1), input_data_4d, layout_4d, torch_kCUDA)
    call torch_tensor_from_array(output_tensor(1), prediction_4d, layout_4d, torch_kCPU)

    ! Load ML model
    call torch_model_load(model, trim(script_model_path) // "/nlgw_unet_gpu_scripted.pt", device_type=torch_kCUDA, device_index=0)

    ! Infer
    do i = 1, num_loops
      call torch_model_forward(model, input_tensor, output_tensor)
    end do

    print *, 'max diff = maxval(abs(prediction - ref_data))'
    print *, maxval(abs(prediction_4d - ref_data_4d))

    success = assert_allclose_real32_4d(prediction_4d, ref_data_4d, "check", 1e-5)
  end if

contains
  subroutine check(status)
  use netcdf

    integer, intent ( in) :: status

    if(status /= nf90_noerr) then 
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check

  subroutine read_dataset_from_nc_2d(dataset, filename)
  use netcdf

  character (len = *), intent(in) :: filename
  real(kind=sp), allocatable, intent(out) :: dataset(:, :)

  real(kind=sp), allocatable :: temp(:, :)

  integer :: dimid_0, dimid_1
  integer :: dim_0, dim_1

  ! This will be the netCDF ID for the file and data variable.
  integer :: ncid, varid

  ! Loop indexes, and error handling.
  integer :: x, y

  print *, 'reading file :: ', filename

  call check( nf90_open(filename, NF90_NOWRITE, ncid) )

  call check( nf90_inq_dimid(ncid, "dim_0", dimid_0) )
  call check( nf90_inq_dimid(ncid, "dim_1", dimid_1) )

  call check( nf90_inquire_dimension(ncid, dimid_0, len = dim_0) )
  call check( nf90_inquire_dimension(ncid, dimid_1, len = dim_1) )

  allocate(temp(dim_1, dim_0))
  allocate(dataset(dim_0, dim_1))

  call check( nf90_inq_varid(ncid, "__xarray_dataarray_variable__", varid) )
  call check( nf90_get_var(ncid, varid, temp) )

  dataset = transpose(temp)

  print *, "dataset shape :: ", shape(dataset)

  ! Close the file, freeing all resources.
  call check( nf90_close(ncid) )
  end subroutine read_dataset_from_nc_2d

  subroutine read_dataset_from_nc_4d(dataset, filename)
  use netcdf

  character (len = *), intent(in) :: filename
  real(kind=sp), allocatable, intent(out) :: dataset(:, :, :, :)

  real(kind=sp), allocatable :: temp(:, :, :, :)

  integer :: dimid_0, dimid_1, dimid_2, dimid_3
  integer :: dim_0, dim_1, dim_2, dim_3

  ! This will be the netCDF ID for the file and data variable.
  integer :: ncid, varid

  ! Loop indexes, and error handling.
  integer :: x, y

  print *, 'reading file :: ', filename

  call check( nf90_open(filename, NF90_NOWRITE, ncid) )

  call check( nf90_inq_dimid(ncid, "dim_0", dimid_0) )
  call check( nf90_inq_dimid(ncid, "dim_1", dimid_1) )
  call check( nf90_inq_dimid(ncid, "dim_2", dimid_2) )
  call check( nf90_inq_dimid(ncid, "dim_3", dimid_3) )

  call check( nf90_inquire_dimension(ncid, dimid_0, len = dim_0) )
  call check( nf90_inquire_dimension(ncid, dimid_1, len = dim_1) )
  call check( nf90_inquire_dimension(ncid, dimid_2, len = dim_2) )
  call check( nf90_inquire_dimension(ncid, dimid_3, len = dim_3) )

  allocate(temp(dim_3, dim_2, dim_1, dim_0))
  allocate(dataset(dim_0, dim_1, dim_2, dim_3))

  call check( nf90_inq_varid(ncid, "__xarray_dataarray_variable__", varid) )
  call check( nf90_get_var(ncid, varid, temp) )

  dataset = reshape(temp, shape=[dim_0, dim_1, dim_2, dim_3], order=[4,3,2,1])

  print *, "dataset shape :: ", shape(dataset)

  ! Close the file, freeing all resources.
  call check( nf90_close(ncid) )
  end subroutine read_dataset_from_nc_4d

  !> Print the result of a test to the terminal
  subroutine test_print(test_name, message, test_pass)

    character(len=*), intent(in) :: test_name  !! Name of the test being run
    character(len=*), intent(in) :: message    !! Message to print
    logical, intent(in) :: test_pass           !! Result of the assertion

    character(len=15) :: report

    if (test_pass) then
      report = char(27)//'[32m'//'PASSED'//char(27)//'[0m'
    else
      report = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    end if
    write(*, '(A, " :: [", A, "] ", A)') report, trim(test_name), trim(message)
  end subroutine test_print

  !> Asserts that two real32-valued 2D arrays coincide to a given relative tolerance
  function assert_allclose_real32_2d(got, expect, test_name, rtol, print_result) result(test_pass)

    character(len=*), intent(in) :: test_name           !! Name of the test being run
    real(kind=sp), intent(in), dimension(:,:) :: got    !! The array of values to be tested
    real(kind=sp), intent(in), dimension(:,:) :: expect !! The array of expected values
    real(kind=sp), intent(in), optional :: rtol         !! Optional relative tolerance (defaults to 1e-5)
    logical, intent(in), optional :: print_result       !! Optionally print test result to screen (defaults to .true.)

    logical :: test_pass  !! Did the assertion pass?

    character(len=80) :: message

    real(kind=sp) :: relative_error
    real(kind=sp) :: rtol_value
    integer :: shape_error
    logical :: print_result_value

    if (.not. present(rtol)) then
      rtol_value = 1.0e-5
    else
      rtol_value = rtol
    end if

    if (.not. present(print_result)) then
      print_result_value = .true.
    else
      print_result_value = print_result
    end if

    ! Check the shapes of the arrays match
    shape_error = maxval(abs(shape(got) - shape(expect)))
    test_pass = (shape_error == 0)

    if (test_pass) then
      test_pass = all(abs(got - expect) <= rtol_value * abs(expect))
      if (print_result_value) then
        write(message,'("relative tolerance = ", E11.4)') rtol_value
        call test_print(test_name, message, test_pass)
      end if
    else if (print_result_value) then
      call test_print(test_name, "Arrays have mismatching shapes.", test_pass)
    endif

  end function assert_allclose_real32_2d

  !> Asserts that two real32-valued 4D arrays coincide to a given relative tolerance
  function assert_allclose_real32_4d(got, expect, test_name, rtol, print_result) result(test_pass)

    character(len=*), intent(in) :: test_name               !! Name of the test being run
    real(kind=sp), intent(in), dimension(:,:,:,:) :: got    !! The array of values to be tested
    real(kind=sp), intent(in), dimension(:,:,:,:) :: expect !! The array of expected values
    real(kind=sp), intent(in), optional :: rtol             !! Optional relative tolerance (defaults to 1e-5)
    logical, intent(in), optional :: print_result           !! Optionally print test result to screen (defaults to .true.)

    logical :: test_pass  !! Did the assertion pass?

    character(len=80) :: message

    real(kind=sp) :: relative_error
    real(kind=sp) :: rtol_value
    integer :: shape_error
    logical :: print_result_value

    if (.not. present(rtol)) then
      rtol_value = 1.0e-5
    else
      rtol_value = rtol
    end if

    if (.not. present(print_result)) then
      print_result_value = .true.
    else
      print_result_value = print_result
    end if

    ! Check the shapes of the arrays match
    shape_error = maxval(abs(shape(got) - shape(expect)))
    test_pass = (shape_error == 0)

    if (test_pass) then
      test_pass = all(abs(got - expect) <= rtol_value * abs(expect))
      if (print_result_value) then
        write(message,'("relative tolerance = ", E11.4)') rtol_value
        call test_print(test_name, message, test_pass)
      end if
    else if (print_result_value) then
      call test_print(test_name, "Arrays have mismatching shapes.", test_pass)
    endif

  end function assert_allclose_real32_4d

end program infer
